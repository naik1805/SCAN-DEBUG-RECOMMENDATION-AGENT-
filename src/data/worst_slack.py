"""
Worst Slack KPI (Timing Debug).

Inputs: same at-speed timing evidence as Timing Violations
  (fail @ high-frequency timing set, pass @ slower set + optional diagnosis slack ps).

Output:
  Fails at 800MHz, passes at 650MHz — ~19% frequency margin proxy; worst slack −47 ps

Frequency margin proxy = (f_fail − f_pass) / f_fail × 100.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.data.paths import (
    COMPILED_DATASET_PATH,
    STIL_PATH,
    WORST_SLACK_PATH,
)
from src.data.timing_violations import build_timing_violation_results

_ROWS_CACHE: Optional[Tuple[str, List[Dict[str, Any]]]] = None
_SUMMARY_CACHE: Optional[Tuple[str, Dict[str, Any]]] = None


def _inputs_fingerprint() -> str:
    parts: List[str] = ["v1-worst-slack-freq-margin"]
    for path in (COMPILED_DATASET_PATH, STIL_PATH):
        if path and os.path.exists(path):
            parts.append(f"{path}:{os.path.getmtime(path)}")
    return "|".join(parts)


def frequency_margin_pct(fail_mhz: float, pass_mhz: float) -> float:
    """(f_fail − f_pass) / f_fail × 100 — e.g. 800 vs 650 → ~19%."""
    if fail_mhz <= 0:
        return 0.0
    raw = (fail_mhz - pass_mhz) / fail_mhz * 100.0
    # Product examples use whole-percent "~19%" style
    return float(round(raw))


def _fmt_mhz(mhz: float) -> str:
    if abs(mhz - int(mhz)) < 1e-9:
        return f"{int(mhz)}"
    return f"{mhz:g}"


def _fmt_slack_ps(slack: Optional[float]) -> Optional[str]:
    if slack is None:
        return None
    # Prefer integer display when whole ps
    if abs(slack - round(slack)) < 1e-9:
        return f"{int(round(slack))} ps"
    return f"{slack:g} ps"


def format_worst_slack_result(
    *,
    fail_mhz: float,
    pass_mhz: float,
    margin_pct: float,
    worst_slack_ps: Optional[float] = None,
) -> str:
    """
    Product template:
      Fails at 800MHz, passes at 650MHz — ~19% frequency margin proxy
    Append worst slack in ps when available.
    """
    head = (
        f"Fails at {_fmt_mhz(fail_mhz)}MHz, passes at {_fmt_mhz(pass_mhz)}MHz — "
        f"~{margin_pct:g}% frequency margin proxy"
    )
    slack_txt = _fmt_slack_ps(worst_slack_ps)
    if slack_txt is not None:
        return f"{head}; worst slack {slack_txt}"
    return head


def format_worst_slack_kpi_value(
    *,
    margin_pct: float,
    worst_slack_ps: Optional[float] = None,
) -> str:
    """Compact card value: '-47ps | ~50%' or '~50%'."""
    slack_txt = _fmt_slack_ps(worst_slack_ps)
    if slack_txt is not None:
        compact = slack_txt.replace(" ", "")
        return f"{compact} | ~{margin_pct:g}%"
    return f"~{margin_pct:g}%"


def _row_from_violation(v: Dict[str, Any]) -> Dict[str, Any]:
    fail_mhz = float(v.get("fastFrequencyMhz") or 0)
    pass_mhz = float(v.get("slowFrequencyMhz") or 0)
    slack = v.get("worstSlackPs")
    try:
        slack_f = float(slack) if slack is not None else None
    except (TypeError, ValueError):
        slack_f = None
    margin = frequency_margin_pct(fail_mhz, pass_mhz)
    result = format_worst_slack_result(
        fail_mhz=fail_mhz,
        pass_mhz=pass_mhz,
        margin_pct=margin,
        worst_slack_ps=slack_f,
    )
    return {
        "result": result,
        "patternId": v.get("patternId"),
        "patternLabel": v.get("patternLabel"),
        "lotId": v.get("lotId"),
        "dieLabel": v.get("dieLabel"),
        "timingChain": v.get("timingChain"),
        "timingFlop": v.get("timingFlop"),
        "kind": v.get("kind"),
        "classification": v.get("classification"),
        "worstSlackPs": slack_f,
        "setupSlackPs": v.get("setupSlackPs"),
        "holdSlackPs": v.get("holdSlackPs"),
        "failFrequencyMhz": fail_mhz,
        "passFrequencyMhz": pass_mhz,
        "frequencyMarginPct": margin,
        "fastTimingSet": v.get("fastTimingSet"),
        "slowTimingSet": v.get("slowTimingSet"),
        "captureEdgeSpacingNs": v.get("captureEdgeSpacingNs"),
        "nearMinimumMargin": v.get("nearMinimumMargin"),
        "testMode": v.get("testMode"),
    }


def _build_summary_and_rows(
    cases: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    violations = build_timing_violation_results(cases)
    rows = [_row_from_violation(v) for v in violations]

    # Worst = most negative slack; tie-break tighter (smaller) frequency margin
    rows.sort(
        key=lambda r: (
            float(r.get("worstSlackPs") if r.get("worstSlackPs") is not None else 0),
            float(r.get("frequencyMarginPct") or 0),
            str(r.get("patternId") or ""),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    if rows:
        top = rows[0]
        summary = {
            "worstSlackPs": top.get("worstSlackPs"),
            "failFrequencyMhz": top.get("failFrequencyMhz"),
            "passFrequencyMhz": top.get("passFrequencyMhz"),
            "frequencyMarginPct": top.get("frequencyMarginPct"),
            "result": top.get("result"),
            "kpiValue": format_worst_slack_kpi_value(
                margin_pct=float(top.get("frequencyMarginPct") or 0),
                worst_slack_ps=top.get("worstSlackPs"),
            ),
            "patternLabel": top.get("patternLabel"),
            "lotId": top.get("lotId"),
            "dieLabel": top.get("dieLabel"),
            "count": len(rows),
        }
    else:
        summary = {
            "worstSlackPs": None,
            "failFrequencyMhz": None,
            "passFrequencyMhz": None,
            "frequencyMarginPct": None,
            "result": "N/A",
            "kpiValue": "N/A",
            "count": 0,
        }
    return summary, rows


def build_worst_slack_results(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    global _ROWS_CACHE, _SUMMARY_CACHE
    fingerprint = _inputs_fingerprint()
    if _ROWS_CACHE is not None and _ROWS_CACHE[0] == fingerprint:
        return _ROWS_CACHE[1]

    persisted = _load_persisted(fingerprint)
    if persisted is not None:
        rows, summary = persisted
        _ROWS_CACHE = (fingerprint, rows)
        _SUMMARY_CACHE = (fingerprint, summary)
        return rows

    summary, rows = _build_summary_and_rows(cases)
    _persist(fingerprint, rows, summary)
    _ROWS_CACHE = (fingerprint, rows)
    _SUMMARY_CACHE = (fingerprint, summary)
    return rows


def get_worst_slack_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    global _SUMMARY_CACHE
    fingerprint = _inputs_fingerprint()
    if _SUMMARY_CACHE is not None and _SUMMARY_CACHE[0] == fingerprint:
        return _SUMMARY_CACHE[1]
    build_worst_slack_results(cases)
    if _SUMMARY_CACHE is not None and _SUMMARY_CACHE[0] == fingerprint:
        return _SUMMARY_CACHE[1]
    return {}


def worst_slack_kpi_value(cases: List[Dict[str, Any]]) -> str:
    summary = get_worst_slack_summary(cases)
    return str(summary.get("kpiValue") or "N/A")


def count_worst_slack_rows(cases: List[Dict[str, Any]]) -> int:
    return len(build_worst_slack_results(cases))


def warm_worst_slack_cache(cases: Optional[List[Dict[str, Any]]] = None) -> int:
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return 0
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if isinstance(cases, dict):
            cases = cases.get("cases") or cases.get("dataset") or []
    return len(build_worst_slack_results(cases or []))


def clear_worst_slack_caches() -> None:
    global _ROWS_CACHE, _SUMMARY_CACHE
    _ROWS_CACHE = None
    _SUMMARY_CACHE = None


def _load_persisted(
    fingerprint: str,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    if not os.path.exists(WORST_SLACK_PATH):
        return None
    try:
        with open(WORST_SLACK_PATH, encoding="utf-8") as f:
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
        os.makedirs(os.path.dirname(WORST_SLACK_PATH), exist_ok=True)
        with open(WORST_SLACK_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "count": len(rows),
                    "summary": summary,
                    "rows": rows,
                },
                f,
                indent=2,
            )
    except OSError:
        pass
