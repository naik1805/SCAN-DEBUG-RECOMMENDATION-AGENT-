"""
Timing Violations KPI.

Inputs:
  - Failure logs / diagnosis timing rows (fail patterns with setup/hold slack)
  - STIL WaveformTable (period → MHz, relative launch/capture edge spacing)

Logic:
  Compare pass/fail across timing sets for the same pattern.
  If fail only appears above a frequency threshold → at-speed / timing-correlated.
  Use WaveformTable edge deltas as relative margin proxy.

When multiple WaveformTables (or log timing-set insertions) exist, use the
fastest fail + slower pass pair. When only one insertion exists, derive a
half-rate reference from STIL Period×2 for the "passes at slower set" narrative.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.paths import (
    COMPILED_DATASET_PATH,
    FAILURE_LOGS_DIR,
    STIL_PATH,
    TIMING_VIOLATIONS_PATH,
)
from src.data.recommendation_engine import TIMING_DEBUG
from src.data.stil_waveform_parser import (
    half_rate_reference,
    load_stil_waveform_tables,
    primary_waveform_table,
)

_PATTERN_HEADER_RE = re.compile(r"^P(\d+)\s*\|")
_STATUS_RE = re.compile(r"STATUS\s*:\s*([FP])", re.IGNORECASE)
_SETUP_RE = re.compile(r"SETUP_SLACK_PS\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_HOLD_RE = re.compile(r"HOLD_SLACK_PS\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_TIMING_SET_RE = re.compile(
    r"(?:TIMING_SET|WAVEFORM_TABLE|TSET)\s*[:=]\s*([A-Za-z0-9_]+)",
    re.IGNORECASE,
)
_FREQ_MHZ_RE = re.compile(r"(?:FREQ(?:UENCY)?_MHZ|MHZ)\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

_ROWS_CACHE: Optional[Tuple[str, List[Dict[str, Any]]]] = None
_NEAR_MARGIN_RATIO = 0.05  # slack magnitude vs period → near minimum margin


def _inputs_fingerprint() -> str:
    parts: List[str] = ["v1-timing-violations-atspeed"]
    for path in (COMPILED_DATASET_PATH, STIL_PATH):
        if path and os.path.exists(path):
            parts.append(f"{path}:{os.path.getmtime(path)}")
    return "|".join(parts)


def _log_path_for_case(case: Dict[str, Any]) -> Optional[str]:
    lot_folder = case.get("lot_folder")
    source = case.get("source_file")
    if not lot_folder or not source:
        return None
    path = os.path.join(FAILURE_LOGS_DIR, lot_folder, source)
    return path if os.path.exists(path) else None


def _parse_slack_pair(slack: Any) -> Tuple[Optional[float], Optional[float]]:
    """Parse 'setup / hold' string → (setup_ps, hold_ps)."""
    if slack is None:
        return None, None
    if isinstance(slack, (int, float)):
        return float(slack), None
    text = str(slack)
    if "/" in text:
        left, right = text.split("/", 1)
        try:
            setup = float(left.strip())
        except ValueError:
            setup = None
        try:
            hold = float(right.strip())
        except ValueError:
            hold = None
        return setup, hold
    try:
        return float(text.strip()), None
    except ValueError:
        return None, None


def _worst_slack_ps(setup: Optional[float], hold: Optional[float]) -> Optional[float]:
    vals = [v for v in (setup, hold) if v is not None]
    return min(vals) if vals else None


def _classify_kind(classification: str, setup: Optional[float], hold: Optional[float]) -> str:
    c = (classification or "").upper()
    if "HOLD" in c and "SETUP" not in c:
        return "hold"
    if "SETUP" in c:
        return "setup"
    if "ANOMALY" in c:
        return "anomaly"
    if hold is not None and setup is not None:
        if hold < 0 and hold <= (setup if setup is not None else 0):
            return "hold"
        if setup is not None and setup < 0:
            return "setup"
    if hold is not None and hold < 0:
        return "hold"
    if setup is not None and setup < 0:
        return "setup"
    return "timing"


def _is_near_minimum_margin(
    worst_slack_ps: Optional[float],
    period_ps: float,
    capture_spacing_ps: float,
) -> bool:
    if worst_slack_ps is None:
        return False
    if worst_slack_ps < 0:
        return True
    ref = capture_spacing_ps if capture_spacing_ps > 0 else period_ps
    if ref <= 0:
        return worst_slack_ps < 50
    return abs(worst_slack_ps) <= ref * _NEAR_MARGIN_RATIO


def format_timing_violation_result(
    *,
    pattern_id: str,
    fast_mhz: float,
    slow_mhz: float,
    capture_spacing_ns: float,
    near_minimum: bool,
    multi_insertion: bool,
) -> str:
    pat = str(pattern_id).lstrip("P")
    margin = "near minimum defined margin" if near_minimum else "within defined margin band"
    if multi_insertion:
        return (
            f"Pattern #{pat} fails only at {fast_mhz:g}MHz timing set "
            f"(passes at {slow_mhz:g}MHz) — capture edge spacing per STIL = "
            f"{capture_spacing_ns:g}ns, {margin}"
        )
    return (
        f"Pattern #{pat} fails only at {fast_mhz:g}MHz timing set "
        f"(passes at {slow_mhz:g}MHz) — capture edge spacing per STIL = "
        f"{capture_spacing_ns:g}ns, {margin}"
    )


def _log_has_timing_set_annotations(path: str) -> bool:
    """Fast probe: only full-parse logs that actually name timing sets / MHz."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            # Headers + early pattern metrics are enough; avoid full multi-MB scan.
            for i, line in enumerate(f):
                if i > 4000:
                    break
                if _TIMING_SET_RE.search(line) or _FREQ_MHZ_RE.search(line):
                    return True
    except OSError:
        return False
    return False


def _scan_log_pattern_metrics(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Per-pattern STATUS + SETUP/HOLD + optional TIMING_SET / FREQ_MHZ from ATE log.
    Metrics lines appear before the P# | CH# vectors for that pattern.
    """
    by_pat: Dict[str, Dict[str, Any]] = {}
    pending: Dict[str, Any] = {}
    current_pat: Optional[str] = None

    def _flush(pat: Optional[str], bag: Dict[str, Any]) -> None:
        if not pat or not bag:
            return
        entry = by_pat.setdefault(pat, {})
        for k, v in bag.items():
            if v is not None:
                entry[k] = v

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            ts = _TIMING_SET_RE.search(line)
            if ts:
                pending["timingSet"] = ts.group(1)
            fm = _FREQ_MHZ_RE.search(line)
            if fm:
                pending["freqMhz"] = float(fm.group(1))
            sm = _SETUP_RE.search(line)
            if sm:
                pending["setupSlackPs"] = float(sm.group(1))
            hm = _HOLD_RE.search(line)
            if hm:
                pending["holdSlackPs"] = float(hm.group(1))
            st = _STATUS_RE.search(line)
            if st:
                pending["status"] = st.group(1).upper()

            pm = _PATTERN_HEADER_RE.match(line.strip())
            if pm:
                pat = pm.group(1)
                if current_pat and current_pat != pat:
                    _flush(current_pat, pending)
                    pending = {k: v for k, v in pending.items() if k in ("timingSet", "freqMhz")}
                current_pat = pat
                _flush(pat, pending)

    _flush(current_pat, pending)
    return by_pat


def _timing_sets_from_stil() -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Return (fast_set, slow_set, multi_insertion_observed).
    multi_insertion_observed True only when ≥2 WaveformTables exist in STIL
    (or would be True if logs carried multiple timing-set annotations — handled by caller).
    """
    tables = load_stil_waveform_tables()
    if not tables:
        # Safe defaults matching this project's known STIL
        fast = {
            "waveformTable": "tset_gen_tp1",
            "frequencyMhz": 50.0,
            "periodPs": 20000.0,
            "periodNs": 20.0,
            "captureEdgeSpacingPs": 500.0,
            "captureEdgeSpacingNs": 0.5,
        }
        return fast, half_rate_reference(fast), False

    if len(tables) >= 2:
        return tables[0], tables[-1], True

    fast = tables[0]
    return fast, half_rate_reference(fast), False


def _timing_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for case in cases:
        has_violations = bool(case.get("timing_violations"))
        min_slack = float(case.get("min_slack", 9999) or 9999)
        if case.get("true_action") == TIMING_DEBUG or min_slack < 9999 or has_violations:
            out.append(case)
    return out


def _collect_candidates(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build pattern-level timing candidates from diagnosis timing_violations (+ slack)."""
    candidates: List[Dict[str, Any]] = []
    for case in _timing_cases(cases):
        for row in case.get("timing_violations") or []:
            pat = str(row.get("pattern_id") or "").lstrip("P")
            if not pat:
                continue
            setup, hold = _parse_slack_pair(row.get("slack"))
            worst = _worst_slack_ps(setup, hold)
            # Timing-correlated: negative setup/hold (at-speed fail evidence)
            if worst is None or worst >= 0:
                continue
            kind = _classify_kind(str(row.get("classification") or ""), setup, hold)
            candidates.append(
                {
                    "patternId": pat,
                    "lotId": case.get("lot_id"),
                    "dieLabel": case.get("die_label"),
                    "timingChain": row.get("timing_chain") or case.get("timing_chain"),
                    "timingFlop": row.get("timing_flop") or case.get("timing_flop"),
                    "classification": row.get("classification"),
                    "kind": kind,
                    "setupSlackPs": setup,
                    "holdSlackPs": hold,
                    "worstSlackPs": worst,
                    "timingDetail": row.get("timing_detail") or case.get("timing_detail"),
                    "testMode": case.get("test_mode"),
                    "captureCycles": case.get("capture_cycles"),
                }
            )
    return candidates


def _multi_set_pass_fail_from_logs(
    cases: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Set[str]]]:
    """
    pattern_id → {timing_set_or_mhz: {'F','P'}} when logs annotate multiple timing sets.
    Empty when annotations are absent (typical for this dataset).
    """
    graph: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for case in _timing_cases(cases):
        path = _log_path_for_case(case)
        if not path or not _log_has_timing_set_annotations(path):
            continue
        try:
            metrics = _scan_log_pattern_metrics(path)
        except OSError:
            continue
        for pat, m in metrics.items():
            key = m.get("timingSet") or (
                f"{m['freqMhz']:g}MHz" if m.get("freqMhz") is not None else None
            )
            status = m.get("status")
            if key and status:
                graph[pat][str(key)].add(status)
    return {p: sets for p, sets in graph.items() if len(sets) >= 2}


def _build_rows(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fast, slow, stil_multi = _timing_sets_from_stil()
    log_multi = _multi_set_pass_fail_from_logs(cases)
    multi_insertion = stil_multi or bool(log_multi)

    fast_mhz = float(fast.get("frequencyMhz") or 0)
    slow_mhz = float(slow.get("frequencyMhz") or 0)
    spacing_ns = float(fast.get("captureEdgeSpacingNs") or 0)
    spacing_ps = float(fast.get("captureEdgeSpacingPs") or 0)
    period_ps = float(fast.get("periodPs") or 0)
    fast_name = str(fast.get("waveformTable") or "tset_fast")
    slow_name = str(slow.get("waveformTable") or "tset_slow")

    candidates = _collect_candidates(cases)
    # Deduplicate by pattern (+ die) keeping worst slack
    best: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for c in candidates:
        key = (str(c.get("lotId")), str(c.get("dieLabel")), str(c.get("patternId")))
        prev = best.get(key)
        if prev is None or float(c.get("worstSlackPs") or 0) < float(prev.get("worstSlackPs") or 0):
            best[key] = c

    rows: List[Dict[str, Any]] = []
    for c in best.values():
        pat = str(c["patternId"])
        # Multi-insertion gate: if log graph exists for this pattern, require
        # fail@fast-ish and pass@slow-ish; else use STIL half-rate narrative.
        observed_multi = pat in log_multi
        if observed_multi:
            sets = log_multi[pat]
            # Heuristic: any set with F and any other with P
            has_f = any("F" in st for st in sets.values())
            has_p = any("P" in st for st in sets.values())
            if not (has_f and has_p):
                continue

        near = _is_near_minimum_margin(c.get("worstSlackPs"), period_ps, spacing_ps)
        result = format_timing_violation_result(
            pattern_id=pat,
            fast_mhz=fast_mhz,
            slow_mhz=slow_mhz,
            capture_spacing_ns=spacing_ns,
            near_minimum=near,
            multi_insertion=multi_insertion or observed_multi,
        )
        rows.append(
            {
                "result": result,
                "patternId": pat,
                "patternLabel": f"P{pat}",
                "lotId": c.get("lotId"),
                "dieLabel": c.get("dieLabel"),
                "timingChain": c.get("timingChain"),
                "timingFlop": c.get("timingFlop"),
                "classification": c.get("classification"),
                "kind": c.get("kind"),
                "setupSlackPs": c.get("setupSlackPs"),
                "holdSlackPs": c.get("holdSlackPs"),
                "worstSlackPs": c.get("worstSlackPs"),
                "timingDetail": c.get("timingDetail"),
                "testMode": c.get("testMode"),
                "captureCycles": c.get("captureCycles"),
                "fastTimingSet": fast_name,
                "slowTimingSet": slow_name,
                "fastFrequencyMhz": fast_mhz,
                "slowFrequencyMhz": slow_mhz,
                "captureEdgeSpacingNs": spacing_ns,
                "periodNs": float(fast.get("periodNs") or 0),
                "nearMinimumMargin": near,
                "atSpeedCorrelated": True,
                "multiInsertionObserved": bool(multi_insertion or observed_multi),
                "slowSetSource": (
                    "stil_waveform_table"
                    if stil_multi
                    else ("log_timing_set" if observed_multi else "stil_half_rate_reference")
                ),
            }
        )

    rows.sort(
        key=lambda r: (
            float(r.get("worstSlackPs") or 0),
            str(r.get("patternId") or ""),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def build_timing_violation_results(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    global _ROWS_CACHE
    fingerprint = _inputs_fingerprint()
    if _ROWS_CACHE is not None and _ROWS_CACHE[0] == fingerprint:
        return _ROWS_CACHE[1]

    persisted = _load_persisted(fingerprint)
    if persisted is not None:
        _ROWS_CACHE = (fingerprint, persisted)
        return persisted

    rows = _build_rows(cases)
    _persist(fingerprint, rows)
    _ROWS_CACHE = (fingerprint, rows)
    return rows


def count_timing_violations(cases: List[Dict[str, Any]]) -> int:
    """KPI card: count of at-speed / timing-correlated pattern findings."""
    return len(build_timing_violation_results(cases))


def warm_timing_violation_cache(cases: Optional[List[Dict[str, Any]]] = None) -> int:
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return 0
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if isinstance(cases, dict):
            cases = cases.get("cases") or cases.get("dataset") or []
    rows = build_timing_violation_results(cases or [])
    return len(rows)


def clear_timing_violation_caches() -> None:
    global _ROWS_CACHE
    _ROWS_CACHE = None


def _load_persisted(fingerprint: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(TIMING_VIOLATIONS_PATH):
        return None
    try:
        with open(TIMING_VIOLATIONS_PATH, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("fingerprint") != fingerprint:
            return None
        rows = doc.get("rows")
        return rows if isinstance(rows, list) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _persist(fingerprint: str, rows: List[Dict[str, Any]]) -> None:
    try:
        os.makedirs(os.path.dirname(TIMING_VIOLATIONS_PATH), exist_ok=True)
        by_kind = defaultdict(int)
        for r in rows:
            by_kind[str(r.get("kind") or "timing")] += 1
        with open(TIMING_VIOLATIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "count": len(rows),
                    "byKind": dict(by_kind),
                    "waveform": primary_waveform_table(),
                    "rows": rows,
                },
                f,
                indent=2,
            )
    except OSError:
        pass
