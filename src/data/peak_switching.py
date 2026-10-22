"""
Peak Switching KPI (Power Related Debug).

Input proxy: IR_DROP_MV from ATE failure logs (electrical downstream effect of
switching activity — no STIL toggle-count computation needed).

Logic:
  MAX(IR_DROP_MV) across all patterns in the run, with pattern reference,
  compared to the run average.

Output:
  Peak IR-drop: 20mV at Pattern #4521 (vs. avg 8mV across run)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.data.paths import (
    COMPILED_DATASET_PATH,
    FAILURE_LOGS_DIR,
    PEAK_SWITCHING_PATH,
)
from src.data.power_violations import (
    _fmt_num,
    _log_path_for_case,
    _power_die_cases,
    clear_power_violation_caches,
    scan_log_power_metrics,
)

_ROWS_CACHE: Optional[Tuple[str, List[Dict[str, Any]]]] = None
_SUMMARY_CACHE: Optional[Tuple[str, Dict[str, Any]]] = None

# Cap drill-down list of highest-IR patterns.
_MAX_WORKSPACE_ROWS = 100


def _inputs_fingerprint() -> str:
    parts: List[str] = ["v1-peak-switching-ir-proxy"]
    if COMPILED_DATASET_PATH and os.path.exists(COMPILED_DATASET_PATH):
        parts.append(f"{COMPILED_DATASET_PATH}:{os.path.getmtime(COMPILED_DATASET_PATH)}")
    if FAILURE_LOGS_DIR and os.path.isdir(FAILURE_LOGS_DIR):
        try:
            parts.append(f"{FAILURE_LOGS_DIR}:{os.path.getmtime(FAILURE_LOGS_DIR)}")
        except OSError:
            pass
    return "|".join(parts)


def format_peak_switching_result(
    *,
    peak_ir_mv: float,
    pattern_id: str,
    avg_ir_mv: float,
) -> str:
    """
    Product template:
      Peak IR-drop: 20mV at Pattern #4521 (vs. avg 8mV across run)
    """
    pat = str(pattern_id).lstrip("P")
    return (
        f"Peak IR-drop: {_fmt_num(float(peak_ir_mv))}mV at Pattern #{pat} "
        f"(vs. avg {_fmt_num(float(avg_ir_mv))}mV across run)"
    )


def format_peak_switching_kpi_value(*, peak_ir_mv: float, avg_ir_mv: float) -> str:
    """Compact card: '28mV | avg 22mV'."""
    return f"{_fmt_num(float(peak_ir_mv))}mV | avg {_fmt_num(float(avg_ir_mv))}mV"


def _collect_pattern_ir(
    cases: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    One entry per unique pattern ID: keep the highest IR_DROP_MV seen across
    power-scoped dies (worst switching-proxy instance for that pattern).
    """
    best: Dict[str, Dict[str, Any]] = {}
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
            ir = m.get("irDropMv")
            if ir is None:
                continue
            pat = str(m.get("patternId") or "")
            if not pat:
                continue
            ir_f = float(ir)
            prev = best.get(pat)
            if prev is None or ir_f > float(prev.get("irDropMv") or 0):
                best[pat] = {
                    "patternId": pat,
                    "patternLabel": f"P{pat}",
                    "lotId": lot,
                    "dieLabel": die,
                    "irDropMv": ir_f,
                    "thermalC": m.get("thermalC"),
                    "status": str(m.get("status") or "P").upper(),
                    "testMode": case.get("test_mode"),
                    "defectType": case.get("defect_type"),
                    "totalPatternsInRun": 1000,
                }
    return best


def _build_summary_and_rows(
    cases: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    by_pat = _collect_pattern_ir(cases)
    if not by_pat:
        summary = {
            "peakIrDropMv": None,
            "avgIrDropMv": None,
            "patternId": None,
            "patternLabel": None,
            "result": "N/A",
            "kpiValue": "N/A",
            "patternCount": 0,
            "count": 0,
        }
        return summary, []

    values = [float(r["irDropMv"]) for r in by_pat.values()]
    avg_ir = round(sum(values) / len(values), 1) if values else 0.0
    # Prefer one decimal when not whole; _fmt_num handles display.

    rows: List[Dict[str, Any]] = []
    for pat, base in by_pat.items():
        ir = float(base["irDropMv"])
        result = format_peak_switching_result(
            peak_ir_mv=ir,
            pattern_id=pat,
            avg_ir_mv=avg_ir,
        )
        # Per-row text still compares that pattern's IR to run avg (leaderboard style);
        # global peak uses the same template with MAX.
        rows.append(
            {
                **base,
                "avgIrDropMv": avg_ir,
                "deltaVsAvgMv": round(ir - avg_ir, 1),
                "result": result,
                "isPeak": False,
                "recommendedAction": "CHECK_IR_DROP_DURING_CAPTURE",
                "recommendDetail": (
                    f"IR_DROP_MV is the switching-activity proxy; pattern IR "
                    f"{_fmt_num(ir)}mV vs run avg {_fmt_num(avg_ir)}mV"
                ),
            }
        )

    rows.sort(
        key=lambda r: (
            -float(r.get("irDropMv") or 0),
            int(str(r.get("patternId") or "0") or 0),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    peak = rows[0]
    peak["isPeak"] = True
    peak_ir = float(peak["irDropMv"])
    peak_result = format_peak_switching_result(
        peak_ir_mv=peak_ir,
        pattern_id=str(peak.get("patternId") or ""),
        avg_ir_mv=avg_ir,
    )
    peak["result"] = peak_result

    # Non-peak rows: "IR-drop: XmV at Pattern #N (vs. avg YmV across run)"
    for row in rows[1:]:
        pat = str(row.get("patternId") or "")
        ir = float(row.get("irDropMv") or 0)
        row["result"] = (
            f"IR-drop: {_fmt_num(ir)}mV at Pattern #{pat.lstrip('P')} "
            f"(vs. avg {_fmt_num(avg_ir)}mV across run)"
        )

    summary = {
        "peakIrDropMv": peak_ir,
        "avgIrDropMv": avg_ir,
        "patternId": peak.get("patternId"),
        "patternLabel": peak.get("patternLabel"),
        "lotId": peak.get("lotId"),
        "dieLabel": peak.get("dieLabel"),
        "result": peak_result,
        "kpiValue": format_peak_switching_kpi_value(peak_ir_mv=peak_ir, avg_ir_mv=avg_ir),
        "patternCount": len(rows),
        "count": len(rows),
        "deltaVsAvgMv": round(peak_ir - avg_ir, 1),
    }
    return summary, rows


def build_peak_switching_results(
    cases: List[Dict[str, Any]],
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
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
            summary, rows = _build_summary_and_rows(cases)
            _persist(fingerprint, rows, summary)
            _ROWS_CACHE = (fingerprint, rows)
            _SUMMARY_CACHE = (fingerprint, summary)

    if limit is not None:
        return rows[: max(0, int(limit))]
    return rows


def build_peak_switching_workspace(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return build_peak_switching_results(cases, limit=_MAX_WORKSPACE_ROWS)


def get_peak_switching_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    global _SUMMARY_CACHE
    fingerprint = _inputs_fingerprint()
    if _SUMMARY_CACHE is not None and _SUMMARY_CACHE[0] == fingerprint:
        return _SUMMARY_CACHE[1]
    build_peak_switching_results(cases)
    if _SUMMARY_CACHE is not None and _SUMMARY_CACHE[0] == fingerprint:
        return _SUMMARY_CACHE[1]
    return {}


def peak_switching_kpi_value(cases: List[Dict[str, Any]]) -> str:
    summary = get_peak_switching_summary(cases)
    return str(summary.get("kpiValue") or "N/A")


def warm_peak_switching_cache(cases: Optional[List[Dict[str, Any]]] = None) -> int:
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return 0
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if isinstance(cases, dict):
            cases = cases.get("cases") or cases.get("dataset") or []
    summary = get_peak_switching_summary(cases or [])
    return int(summary.get("patternCount") or 0)


def clear_peak_switching_caches() -> None:
    global _ROWS_CACHE, _SUMMARY_CACHE
    _ROWS_CACHE = None
    _SUMMARY_CACHE = None
    clear_power_violation_caches()


def _load_persisted(
    fingerprint: str,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    if not os.path.exists(PEAK_SWITCHING_PATH):
        return None
    try:
        with open(PEAK_SWITCHING_PATH, encoding="utf-8") as f:
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
        os.makedirs(os.path.dirname(PEAK_SWITCHING_PATH), exist_ok=True)
        with open(PEAK_SWITCHING_PATH, "w", encoding="utf-8") as f:
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
