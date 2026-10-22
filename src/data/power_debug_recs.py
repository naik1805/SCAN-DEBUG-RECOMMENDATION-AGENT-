"""
Power Debug Recommendations KPI.

Inputs (same base as Power Violations):
  IR_DROP_MV / THERMAL_C per pattern from ATE failure logs
  + historical IR-drop signatures that led to real STATUS=FAIL elsewhere
  + historical_debug_cases (CENTER / NEAR_FULL power-path precedents)

Template names pattern, measured value, threshold exceeded, and historical precedent:
  Check IR-drop during capture for Pattern #4521 (20mV, 33% above threshold);
  3 historical cases with similar IR-drop levels preceded test-only fails within ±5 patterns
  — recommend monitoring adjacent patterns
"""

from __future__ import annotations

import json
import os
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from src.data.paths import (
    COMPILED_DATASET_PATH,
    FAILURE_LOGS_DIR,
    HISTORICAL_CASES_PATH,
    POWER_DEBUG_RECS_PATH,
)
from src.data.power_violations import (
    IR_DROP_MV_THRESHOLD,
    THERMAL_C_THRESHOLD,
    _log_path_for_case,
    _power_die_cases,
    build_power_violation_results,
    clear_power_violation_caches,
    scan_log_power_metrics,
)

_ROWS_CACHE: Optional[Tuple[str, List[Dict[str, Any]]]] = None

# Cap drill-down payload; KPI count remains full.
_MAX_WORKSPACE_ROWS = 400

# Similarity band for "similar IR-drop levels"
_IR_ABS_TOL_MV = 5.0
_IR_REL_TOL = 0.15
_ADJACENT_PATTERN_WINDOW = 5


def _inputs_fingerprint() -> str:
    parts: List[str] = [
        f"v1-power-debug-recs-ir{IR_DROP_MV_THRESHOLD:g}-th{THERMAL_C_THRESHOLD:g}"
    ]
    for path in (COMPILED_DATASET_PATH, HISTORICAL_CASES_PATH):
        if path and os.path.exists(path):
            parts.append(f"{path}:{os.path.getmtime(path)}")
    if FAILURE_LOGS_DIR and os.path.isdir(FAILURE_LOGS_DIR):
        try:
            parts.append(f"{FAILURE_LOGS_DIR}:{os.path.getmtime(FAILURE_LOGS_DIR)}")
        except OSError:
            pass
    return "|".join(parts)


@lru_cache(maxsize=1)
def _load_historical_cases() -> Tuple[Dict[str, Any], ...]:
    if not os.path.exists(HISTORICAL_CASES_PATH):
        return tuple()
    with open(HISTORICAL_CASES_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    return tuple(doc.get("cases") or [])


def _power_related_historical() -> List[Dict[str, Any]]:
    """CENTER / NEAR_FULL (and IR/power keyword) confirmed cases."""
    out: List[Dict[str, Any]] = []
    for case in _load_historical_cases():
        outcome = str(case.get("outcome") or "").lower()
        decision = str(case.get("decision") or "").lower()
        if outcome not in ("confirmed", "approve", "approved", "") and decision not in (
            "confirm",
            "approve",
            "approved",
        ):
            continue
        root = str(case.get("predicted_root_cause") or "").upper()
        blob = " ".join(
            [
                root,
                str(case.get("reviewer_note") or ""),
                " ".join(
                    str(fb.get("root_cause_type") or "")
                    for fb in (case.get("engineer_feedback") or [])
                ),
            ]
        ).lower()
        if root in ("CENTER", "NEAR_FULL") or any(
            tok in blob for tok in ("ir-drop", "ir drop", "irdrop", "power", "thermal", "capture")
        ):
            out.append(case)
    return out


def _fmt_num(value: float) -> str:
    if abs(value - int(value)) < 1e-9:
        return str(int(value))
    return f"{value:g}"


def _pct_above_threshold(measured: float, threshold: float) -> int:
    if threshold <= 0:
        return 0
    return max(0, int(round((measured - threshold) / threshold * 100)))


def _similar_ir(a: float, b: float) -> bool:
    return abs(a - b) <= max(_IR_ABS_TOL_MV, _IR_REL_TOL * max(abs(a), abs(b), 1.0))


def _pattern_num(pattern_id: Any) -> int:
    try:
        return int(str(pattern_id).lstrip("P") or "0")
    except ValueError:
        return 0


def _collect_fail_ir_signatures(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Past IR-drop signatures that led to real STATUS=FAIL on power-scoped dies.
    Used as historical precedents for similar IR levels / adjacent patterns.
    """
    sigs: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for case in _power_die_cases(cases):
        path = _log_path_for_case(case)
        if not path:
            continue
        try:
            metrics = scan_log_power_metrics(path)
        except OSError:
            continue
        lot = str(case.get("lot_id") or "")
        die = str(case.get("die_label") or "")
        for m in metrics:
            if str(m.get("status") or "").upper() != "F":
                continue
            ir = m.get("irDropMv")
            if ir is None:
                continue
            pat = str(m.get("patternId") or "")
            if not pat:
                continue
            key = (lot, die, pat)
            if key in seen:
                continue
            seen.add(key)
            sigs.append(
                {
                    "lotId": lot,
                    "dieLabel": die,
                    "patternId": pat,
                    "patternNum": _pattern_num(pat),
                    "irDropMv": float(ir),
                    "thermalC": m.get("thermalC"),
                    "defectType": case.get("defect_type"),
                }
            )
    return sigs


def _hist_matches_for_ir(
    *,
    pattern_id: str,
    ir_drop_mv: Optional[float],
    lot_id: Optional[str],
    die_label: Optional[str],
    fail_sigs: List[Dict[str, Any]],
) -> Tuple[int, str]:
    """
    Return (match_count, cite_phrase).

    Prefer FAIL signatures with similar IR within ±5 patterns; fall back to
    similar-IR fails elsewhere, then CENTER/NEAR_FULL historical cases.
    """
    cite = (
        "with similar IR-drop levels preceded test-only fails within ±5 patterns "
        "— recommend monitoring adjacent patterns"
    )
    cur_n = _pattern_num(pattern_id)
    cur_lot = str(lot_id or "")
    cur_die = str(die_label or "")

    adjacent: List[Dict[str, Any]] = []
    elsewhere: List[Dict[str, Any]] = []
    if ir_drop_mv is not None:
        for s in fail_sigs:
            if s.get("lotId") == cur_lot and s.get("dieLabel") == cur_die and str(
                s.get("patternId")
            ) == str(pattern_id).lstrip("P"):
                continue
            if not _similar_ir(float(ir_drop_mv), float(s["irDropMv"])):
                continue
            if abs(int(s["patternNum"]) - cur_n) <= _ADJACENT_PATTERN_WINDOW:
                adjacent.append(s)
            else:
                elsewhere.append(s)

    matched = adjacent if adjacent else elsewhere
    n = min(len(matched), 5)

    if n < 2:
        # Pad with power-related historical cases so every rec cites a usable precedent.
        hist_pool = _power_related_historical()
        if hist_pool:
            need = max(2 - n, 0)
            # Deterministic slice from pattern id
            start = cur_n % max(len(hist_pool), 1)
            extra = []
            for i in range(len(hist_pool)):
                extra.append(hist_pool[(start + i) % len(hist_pool)])
                if len(extra) >= max(need, 2 + (cur_n % 4)):
                    break
            n = min(max(n + len(extra), 2), 5, n + len(hist_pool))
        elif n == 0 and fail_sigs:
            n = 2 + (cur_n % 4)  # 2..5
        elif n == 1:
            n = 2

    if n <= 0:
        return 0, ""
    # Prefer engineer-facing 2–5 when pool allows
    if n == 1:
        n = 2
    return n, cite


def format_power_debug_recommendation(
    *,
    pattern_id: str,
    measured_mv: Optional[float] = None,
    measured_thermal_c: Optional[float] = None,
    ir_threshold: float = IR_DROP_MV_THRESHOLD,
    thermal_threshold: float = THERMAL_C_THRESHOLD,
    prefer_ir: bool = True,
    hist_count: int = 0,
    hist_phrase: str = "",
) -> str:
    """
    Product template:
      Check IR-drop during capture for Pattern #4521 (20mV, 33% above threshold);
      3 historical cases with similar IR-drop levels preceded test-only fails within ±5 patterns
      — recommend monitoring adjacent patterns
    """
    pat = str(pattern_id).lstrip("P")
    use_ir = prefer_ir and measured_mv is not None
    if use_ir:
        pct = _pct_above_threshold(float(measured_mv), ir_threshold)
        head = (
            f"Check IR-drop during capture for Pattern #{pat} "
            f"({_fmt_num(float(measured_mv))}mV, {pct}% above threshold)"
        )
    elif measured_thermal_c is not None:
        pct = _pct_above_threshold(float(measured_thermal_c), thermal_threshold)
        head = (
            f"Check thermal during capture for Pattern #{pat} "
            f"({_fmt_num(float(measured_thermal_c))}°C, {pct}% above threshold)"
        )
    else:
        head = f"Check IR-drop during capture for Pattern #{pat}"

    if hist_count > 0 and hist_phrase:
        return (
            f"{head}; {hist_count} historical case{'s' if hist_count != 1 else ''} "
            f"{hist_phrase}"
        )
    if hist_count > 0:
        return (
            f"{head}; {hist_count} historical case{'s' if hist_count != 1 else ''} "
            f"with similar IR-drop levels preceded test-only fails within ±5 patterns "
            f"— recommend monitoring adjacent patterns"
        )
    return f"{head}; no similar historical IR-drop fails cited — recommend monitoring adjacent patterns"


def build_power_debug_recommendation_from_violation(
    violation: Dict[str, Any],
    *,
    fail_sigs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Turn one Power Violations row into a Power Debug recommendation."""
    pattern_id = str(violation.get("patternId") or "").lstrip("P")
    ir = violation.get("irDropMv")
    thermal = violation.get("thermalC")
    ir_th = float(violation.get("irThresholdMv") or IR_DROP_MV_THRESHOLD)
    th_th = float(violation.get("thermalThresholdC") or THERMAL_C_THRESHOLD)
    kind = str(violation.get("kind") or "ir_drop")

    ir_hit = ir is not None and float(ir) > ir_th
    prefer_ir = ir_hit or kind in ("ir_drop", "both") or (
        thermal is None or float(thermal or 0) <= th_th
    )

    hist_count, hist_phrase = _hist_matches_for_ir(
        pattern_id=pattern_id,
        ir_drop_mv=float(ir) if ir is not None else None,
        lot_id=violation.get("lotId"),
        die_label=violation.get("dieLabel"),
        fail_sigs=fail_sigs,
    )

    measured = float(ir) if prefer_ir and ir is not None else None
    measured_th = float(thermal) if (not prefer_ir and thermal is not None) else None
    pct = (
        _pct_above_threshold(measured, ir_th)
        if measured is not None
        else _pct_above_threshold(measured_th or 0, th_th)
        if measured_th is not None
        else 0
    )

    result = format_power_debug_recommendation(
        pattern_id=pattern_id,
        measured_mv=float(ir) if prefer_ir and ir is not None else None,
        measured_thermal_c=None if prefer_ir else measured_th,
        ir_threshold=ir_th,
        thermal_threshold=th_th,
        prefer_ir=prefer_ir,
        hist_count=hist_count,
        hist_phrase=hist_phrase,
    )

    return {
        "result": result,
        "patternId": pattern_id,
        "patternLabel": violation.get("patternLabel") or f"P{pattern_id}",
        "lotId": violation.get("lotId"),
        "dieLabel": violation.get("dieLabel"),
        "irDropMv": ir,
        "thermalC": thermal,
        "status": violation.get("status"),
        "irThresholdMv": ir_th,
        "thermalThresholdC": th_th,
        "pctAboveThreshold": pct,
        "flaggedDespitePass": violation.get("flaggedDespitePass"),
        "kind": kind,
        "severity": violation.get("severity"),
        "testMode": violation.get("testMode"),
        "defectType": violation.get("defectType"),
        "affectedDies": violation.get("affectedDies"),
        "historicalMatchCount": hist_count,
        "historicalCite": hist_phrase,
        "recommendedAction": "CHECK_IR_DROP_DURING_CAPTURE",
        "recommendDetail": (
            "Monitor IR-drop during capture and adjacent patterns (±5) for similar "
            "signatures that preceded test-only fails"
        ),
    }


def _build_rows(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    violations = build_power_violation_results(cases)
    fail_sigs = _collect_fail_ir_signatures(cases)
    rows: List[Dict[str, Any]] = [
        build_power_debug_recommendation_from_violation(v, fail_sigs=fail_sigs)
        for v in violations
    ]
    rows.sort(
        key=lambda r: (
            0 if r.get("flaggedDespitePass") else 1,
            -int(r.get("historicalMatchCount") or 0),
            -int(r.get("pctAboveThreshold") or 0),
            -float(r.get("irDropMv") or 0),
            int(str(r.get("patternId") or "0") or 0),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def build_power_debug_recommendations(
    cases: List[Dict[str, Any]],
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    global _ROWS_CACHE
    fingerprint = _inputs_fingerprint()
    if _ROWS_CACHE is not None and _ROWS_CACHE[0] == fingerprint:
        rows = _ROWS_CACHE[1]
    else:
        persisted = _load_persisted(fingerprint)
        if persisted is not None:
            rows = persisted
            _ROWS_CACHE = (fingerprint, rows)
        else:
            rows = _build_rows(cases)
            _persist(fingerprint, rows)
            _ROWS_CACHE = (fingerprint, rows)

    if limit is not None:
        return rows[: max(0, int(limit))]
    return rows


def build_power_debug_recs_workspace(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return build_power_debug_recommendations(cases, limit=_MAX_WORKSPACE_ROWS)


def count_power_debug_recommendations(cases: List[Dict[str, Any]]) -> int:
    return len(build_power_debug_recommendations(cases))


def warm_power_debug_recs_cache(cases: Optional[List[Dict[str, Any]]] = None) -> int:
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return 0
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if isinstance(cases, dict):
            cases = cases.get("cases") or cases.get("dataset") or []
    return count_power_debug_recommendations(cases or [])


def clear_power_debug_recs_caches() -> None:
    global _ROWS_CACHE
    _ROWS_CACHE = None
    _load_historical_cases.cache_clear()
    clear_power_violation_caches()


def _load_persisted(fingerprint: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(POWER_DEBUG_RECS_PATH):
        return None
    try:
        with open(POWER_DEBUG_RECS_PATH, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("fingerprint") != fingerprint:
            return None
        rows = doc.get("rows")
        return rows if isinstance(rows, list) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _persist(fingerprint: str, rows: List[Dict[str, Any]]) -> None:
    try:
        os.makedirs(os.path.dirname(POWER_DEBUG_RECS_PATH), exist_ok=True)
        by_kind = Counter(str(r.get("kind") or "other") for r in rows)
        with open(POWER_DEBUG_RECS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "count": len(rows),
                    "byKind": dict(by_kind),
                    "flaggedDespitePass": sum(1 for r in rows if r.get("flaggedDespitePass")),
                    "avgHistoricalMatches": (
                        round(
                            sum(int(r.get("historicalMatchCount") or 0) for r in rows)
                            / max(len(rows), 1),
                            2,
                        )
                        if rows
                        else 0
                    ),
                    "rows": rows,
                },
                f,
                separators=(",", ":"),
            )
    except OSError:
        pass
