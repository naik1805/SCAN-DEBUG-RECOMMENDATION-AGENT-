"""
Power Violations KPI.

Inputs (ATE failure logs, all patterns in the run):
  IR_DROP_MV, THERMAL_C per pattern (+ STATUS)

Logic:
  Flag pattern if IR_DROP_MV > IR threshold (25mV on this dataset; product e.g. 15mV)
  or THERMAL_C > thermal threshold (60°C on this dataset; product e.g. 55°C),
  even when STATUS=PASS (marginal patterns often pass but are close to failing).

Output:
  Pattern #4521: IR_DROP = 28mV (threshold 25mV) — flagged despite PASS status
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from src.data.paths import (
    COMPILED_DATASET_PATH,
    FAILURE_LOGS_DIR,
    POWER_VIOLATIONS_PATH,
)
from src.data.recommendation_engine import POWER_DEBUG
from src.config import get_settings

# Thresholds from env (see IR_DROP_MV_THRESHOLD / THERMAL_C_THRESHOLD).
IR_DROP_MV_THRESHOLD = get_settings().ir_drop_mv_threshold
THERMAL_C_THRESHOLD = get_settings().thermal_c_threshold

# Cap drill-down payload size; KPI count remains full.
_MAX_WORKSPACE_ROWS = get_settings().power_workspace_rows

_PATTERN_RE = re.compile(r"^P(\d+)\s*\|")
_STATUS_RE = re.compile(r"STATUS\s*:\s*([FP])", re.IGNORECASE)
_IR_RE = re.compile(r"IR_DROP_MV\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_THERMAL_RE = re.compile(r"THERMAL_C\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_TRANSITION_RE = re.compile(r"TRANSITION_FAULTS\s*:\s*(-?\d+)", re.IGNORECASE)

_ROWS_CACHE: Optional[Tuple[str, List[Dict[str, Any]]]] = None
_SUMMARY_CACHE: Optional[Tuple[str, Dict[str, Any]]] = None


def _inputs_fingerprint() -> str:
    parts: List[str] = [
        f"v3-power-dies-ir{IR_DROP_MV_THRESHOLD:g}-th{THERMAL_C_THRESHOLD:g}"
    ]
    if COMPILED_DATASET_PATH and os.path.exists(COMPILED_DATASET_PATH):
        parts.append(f"{COMPILED_DATASET_PATH}:{os.path.getmtime(COMPILED_DATASET_PATH)}")
    # Fingerprint fail-log tree lightly via folder mtimes (avoid hashing every file).
    if FAILURE_LOGS_DIR and os.path.isdir(FAILURE_LOGS_DIR):
        try:
            parts.append(f"{FAILURE_LOGS_DIR}:{os.path.getmtime(FAILURE_LOGS_DIR)}")
        except OSError:
            pass
    return "|".join(parts)


def _log_path_for_case(case: Dict[str, Any]) -> Optional[str]:
    lot_folder = case.get("lot_folder")
    source = case.get("source_file")
    if not lot_folder or not source:
        return None
    path = os.path.join(FAILURE_LOGS_DIR, lot_folder, source)
    return path if os.path.exists(path) else None


def _fmt_num(value: float) -> str:
    if abs(value - int(value)) < 1e-9:
        return str(int(value))
    return f"{value:g}"


def format_power_violation_result(
    *,
    pattern_id: str,
    ir_drop_mv: Optional[float],
    thermal_c: Optional[float],
    status: str,
    ir_threshold: float = IR_DROP_MV_THRESHOLD,
    thermal_threshold: float = THERMAL_C_THRESHOLD,
) -> str:
    """
    Prefer IR citation when IR exceeds threshold (matches product example);
    else cite thermal. Always note PASS when status is PASS.
    """
    pat = str(pattern_id).lstrip("P")
    ir_hit = ir_drop_mv is not None and ir_drop_mv > ir_threshold
    th_hit = thermal_c is not None and thermal_c > thermal_threshold

    if ir_hit and (not th_hit or (ir_drop_mv or 0) - ir_threshold >= (thermal_c or 0) - thermal_threshold):
        head = (
            f"Pattern #{pat}: IR_DROP = {_fmt_num(float(ir_drop_mv))}mV "
            f"(threshold {_fmt_num(ir_threshold)}mV)"
        )
    elif th_hit:
        head = (
            f"Pattern #{pat}: THERMAL = {_fmt_num(float(thermal_c))}°C "
            f"(threshold {_fmt_num(thermal_threshold)}°C)"
        )
    else:
        head = f"Pattern #{pat}: within IR/thermal thresholds"

    if str(status).upper() == "P":
        return f"{head} — flagged despite PASS status"
    return f"{head} — flagged (STATUS=FAIL)"


def scan_log_power_metrics(path: str) -> List[Dict[str, Any]]:
    """
    Parse per-pattern IR_DROP_MV / THERMAL_C / TRANSITION_FAULTS / STATUS from an ATE log.

    Pattern vectors (with STATUS) appear first; PATTERN_METRICS (IR/thermal/TF) follow
    before the next separator / pattern.
    """
    rows: List[Dict[str, Any]] = []
    current_pat: Optional[str] = None
    status = "P"
    ir_drop: Optional[float] = None
    thermal: Optional[float] = None
    transition: Optional[int] = None
    saw_metrics = False

    def _flush() -> None:
        nonlocal current_pat, status, ir_drop, thermal, transition, saw_metrics
        if current_pat is None:
            return
        if ir_drop is None and thermal is None and transition is None:
            return
        rows.append(
            {
                "patternId": current_pat,
                "irDropMv": ir_drop,
                "thermalC": thermal,
                "transitionFaults": transition,
                "status": status,
            }
        )
        current_pat = None
        status = "P"
        ir_drop = None
        thermal = None
        transition = None
        saw_metrics = False

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            pm = _PATTERN_RE.match(line.strip())
            if pm:
                pat = pm.group(1)
                if current_pat is not None and pat != current_pat:
                    # New pattern — flush previous if metrics were captured
                    if (
                        saw_metrics
                        or ir_drop is not None
                        or thermal is not None
                        or transition is not None
                    ):
                        _flush()
                    else:
                        # Vectors without metrics yet shouldn't leave a stale pattern
                        current_pat = None
                        status = "P"
                        ir_drop = None
                        thermal = None
                        transition = None
                        saw_metrics = False
                if current_pat != pat:
                    current_pat = pat
                    status = "P"
                    ir_drop = None
                    thermal = None
                    transition = None
                    saw_metrics = False
                continue

            st = _STATUS_RE.search(line)
            if st and current_pat is not None:
                if st.group(1).upper() == "F":
                    status = "F"

            ir_m = _IR_RE.search(line)
            if ir_m:
                ir_drop = float(ir_m.group(1))
                saw_metrics = True
            th_m = _THERMAL_RE.search(line)
            if th_m:
                thermal = float(th_m.group(1))
                saw_metrics = True
            tf_m = _TRANSITION_RE.search(line)
            if tf_m:
                transition = int(float(tf_m.group(1)))
                saw_metrics = True

            if line.strip().startswith("-----") and current_pat is not None and (
                saw_metrics
                or ir_drop is not None
                or thermal is not None
                or transition is not None
            ):
                _flush()

    if current_pat is not None and (
        saw_metrics or ir_drop is not None or thermal is not None or transition is not None
    ):
        _flush()
    return rows


def _is_violation(
    ir_drop: Optional[float],
    thermal: Optional[float],
    *,
    ir_threshold: float,
    thermal_threshold: float,
) -> bool:
    if ir_drop is not None and ir_drop > ir_threshold:
        return True
    if thermal is not None and thermal > thermal_threshold:
        return True
    return False


def _severity(
    ir_drop: Optional[float],
    thermal: Optional[float],
    *,
    ir_threshold: float,
    thermal_threshold: float,
) -> float:
    excess = 0.0
    if ir_drop is not None and ir_drop > ir_threshold:
        excess += ir_drop - ir_threshold
    if thermal is not None and thermal > thermal_threshold:
        excess += thermal - thermal_threshold
    return excess


def _power_die_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Power Violations scope: fail dies on the Check-IR-Drop path, or CENTER/NEAR_FULL
    defect shapes (same heuristic as POWER_DEBUG action inference).
    """
    out: List[Dict[str, Any]] = []
    for case in cases:
        if not str(case.get("source_file", "")).startswith("fail"):
            continue
        defect = str(case.get("defect_type") or "").upper()
        if case.get("true_action") == POWER_DEBUG or defect in ("CENTER", "NEAR_FULL"):
            out.append(case)
    return out


def _build_rows(
    cases: List[Dict[str, Any]],
    *,
    ir_threshold: float = IR_DROP_MV_THRESHOLD,
    thermal_threshold: float = THERMAL_C_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Return one row per unique pattern ID in the 1000-pattern run.
    If the same pattern trips power thresholds on multiple dies, keep the worst
    IR/thermal instance and record how many dies were affected.
    """
    best_by_pattern: Dict[str, Dict[str, Any]] = {}
    dies_by_pattern: Dict[str, set[str]] = {}

    for case in _power_die_cases(cases):
        path = _log_path_for_case(case)
        if not path:
            continue
        try:
            metrics = scan_log_power_metrics(path)
        except OSError:
            continue

        lot = case.get("lot_id")
        die = case.get("die_label")

        for m in metrics:
            ir_drop = m.get("irDropMv")
            thermal = m.get("thermalC")
            if not _is_violation(
                ir_drop,
                thermal,
                ir_threshold=ir_threshold,
                thermal_threshold=thermal_threshold,
            ):
                continue

            status = str(m.get("status") or "P").upper()
            pat = str(m.get("patternId") or "")
            if not pat:
                continue

            ir_hit = ir_drop is not None and ir_drop > ir_threshold
            th_hit = thermal is not None and thermal > thermal_threshold
            kind = (
                "ir_drop"
                if ir_hit and not th_hit
                else "thermal"
                if th_hit and not ir_hit
                else "both"
            )
            severity = _severity(
                ir_drop,
                thermal,
                ir_threshold=ir_threshold,
                thermal_threshold=thermal_threshold,
            )
            candidate = {
                "patternId": pat,
                "patternLabel": f"P{pat}",
                "lotId": lot,
                "dieLabel": die,
                "irDropMv": ir_drop,
                "thermalC": thermal,
                "status": status,
                "irThresholdMv": ir_threshold,
                "thermalThresholdC": thermal_threshold,
                "flaggedDespitePass": status == "P",
                "kind": kind,
                "severity": severity,
                "testMode": case.get("test_mode"),
                "defectType": case.get("defect_type"),
                "totalPatternsInRun": 1000,
            }
            dies_by_pattern.setdefault(pat, set()).add(f"{lot}/{die}")

            prev = best_by_pattern.get(pat)
            if prev is None or severity > float(prev.get("severity") or 0):
                best_by_pattern[pat] = candidate
            elif severity == float(prev.get("severity") or 0):
                # Prefer PASS-status marginal narrative on ties.
                if candidate["flaggedDespitePass"] and not prev.get("flaggedDespitePass"):
                    best_by_pattern[pat] = candidate

    rows: List[Dict[str, Any]] = []
    for pat, row in best_by_pattern.items():
        out = dict(row)
        affected = sorted(dies_by_pattern.get(pat) or [])
        out["affectedDies"] = len(affected)
        out["affectedDieLabels"] = affected[:8]
        out["result"] = format_power_violation_result(
            pattern_id=pat,
            ir_drop_mv=out.get("irDropMv"),
            thermal_c=out.get("thermalC"),
            status=str(out.get("status") or "P"),
            ir_threshold=ir_threshold,
            thermal_threshold=thermal_threshold,
        )
        rows.append(out)

    rows.sort(
        key=lambda r: (
            # Prefer marginal PASS findings (product narrative) before hard FAILs
            0 if r.get("flaggedDespitePass") else 1,
            -float(r.get("severity") or 0),
            -float(r.get("irDropMv") or 0),
            -float(r.get("thermalC") or 0),
            int(str(r.get("patternId") or "0") or 0),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def build_power_violation_results(
    cases: List[Dict[str, Any]],
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Full ranked findings; optional limit for workspace payloads."""
    global _ROWS_CACHE, _SUMMARY_CACHE
    fingerprint = _inputs_fingerprint()
    if _ROWS_CACHE is not None and _ROWS_CACHE[0] == fingerprint:
        rows = _ROWS_CACHE[1]
    else:
        persisted = _load_persisted(fingerprint)
        if persisted is not None:
            rows, summary = persisted
            _ROWS_CACHE = (fingerprint, rows)
            _SUMMARY_CACHE = (fingerprint, summary)
        else:
            rows = _build_rows(cases)
            summary = _summary_from_rows(rows)
            _persist(fingerprint, rows, summary)
            _ROWS_CACHE = (fingerprint, rows)
            _SUMMARY_CACHE = (fingerprint, summary)

    if limit is not None:
        return rows[: max(0, int(limit))]
    return rows


def build_power_violation_workspace_results(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return build_power_violation_results(cases, limit=_MAX_WORKSPACE_ROWS)


def _summary_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_kind = Counter(str(r.get("kind") or "other") for r in rows)
    despite_pass = sum(1 for r in rows if r.get("flaggedDespitePass"))
    total_patterns = 1000
    count = len(rows)
    return {
        "count": count,
        "totalPatternsInRun": total_patterns,
        "kpiValue": f"{count}/{total_patterns}",
        "flaggedDespitePass": despite_pass,
        "byKind": dict(by_kind),
        "irThresholdMv": IR_DROP_MV_THRESHOLD,
        "thermalThresholdC": THERMAL_C_THRESHOLD,
    }


def get_power_violation_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    global _SUMMARY_CACHE
    fingerprint = _inputs_fingerprint()
    if _SUMMARY_CACHE is not None and _SUMMARY_CACHE[0] == fingerprint:
        return _SUMMARY_CACHE[1]
    build_power_violation_results(cases)
    if _SUMMARY_CACHE is not None and _SUMMARY_CACHE[0] == fingerprint:
        return _SUMMARY_CACHE[1]
    return {}


def count_power_violations(cases: List[Dict[str, Any]]) -> int:
    summary = get_power_violation_summary(cases)
    return int(summary.get("count") or 0)


def power_violations_kpi_value(cases: List[Dict[str, Any]]) -> str:
    summary = get_power_violation_summary(cases)
    return str(summary.get("kpiValue") or "0/1000")


def warm_power_violation_cache(cases: Optional[List[Dict[str, Any]]] = None) -> int:
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return 0
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if isinstance(cases, dict):
            cases = cases.get("cases") or cases.get("dataset") or []
    return count_power_violations(cases or [])


def clear_power_violation_caches() -> None:
    global _ROWS_CACHE, _SUMMARY_CACHE
    _ROWS_CACHE = None
    _SUMMARY_CACHE = None


def _load_persisted(
    fingerprint: str,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    if not os.path.exists(POWER_VIOLATIONS_PATH):
        return None
    try:
        with open(POWER_VIOLATIONS_PATH, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("fingerprint") != fingerprint:
            return None
        rows = doc.get("rows")
        summary = doc.get("summary") or {}
        if not isinstance(rows, list):
            return None
        return rows, summary
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _persist(fingerprint: str, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(POWER_VIOLATIONS_PATH), exist_ok=True)
        with open(POWER_VIOLATIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "count": len(rows),
                    "summary": summary,
                    "rows": rows,
                },
                f,
                separators=(",", ":"),
            )
    except OSError:
        pass
