"""
ATPG Coverage Impact KPI.

Input proxy: failing bitmap / ATE failing-pattern counts associated with
constraint signatures (held pin × fan-out), vs all failing patterns in the dataset.

KPI card (whole-data):
  ~N% of all failing patterns are associated with ATPG constraints — estimate only

Per-signature / per-category drill-down still shows individual shares.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.atpg_constraint_violations import (
    build_constraint_violation_results,
    failing_pattern_channel_pairs,
)
from src.data.paths import (
    COMPILED_DATASET_PATH,
    COVERAGE_IMPACT_PATH,
    FAILURE_LOGS_DIR,
    STIL_PATH,
)

_CATEGORY_ORDER = ("reset", "scan_enable", "clock")
_COVERAGE_ROWS_CACHE: Optional[Tuple[str, List[Dict[str, Any]]]] = None
_COVERAGE_SUMMARY_CACHE: Optional[Tuple[str, Dict[str, Any]]] = None


def _inputs_fingerprint() -> str:
    parts: List[str] = ["v2-overall-coverage-impact"]
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


def _all_failing_pattern_labels(cases: List[Dict[str, Any]]) -> Set[str]:
    """Union of failing pattern IDs across fail-die ATE logs (bitmap proxy)."""
    labels: Set[str] = set()
    for case in cases:
        if not str(case.get("source_file", "")).startswith("fail"):
            continue
        compiled = case.get("failing_patterns")
        if isinstance(compiled, list) and compiled:
            labels.update(str(p) for p in compiled)
            continue
        path = _log_path_for_case(case)
        if not path:
            n = int(case.get("failing_pattern_count") or 0)
            if n > 0:
                labels.update(f"P{i}" for i in range(1, n + 1))
            continue
        for pat, _fanout in failing_pattern_channel_pairs(path):
            labels.add(pat)
    return labels


def _fmt_pct(pct: float) -> float:
    pct = round(pct, 1)
    if pct >= 10 and pct == int(pct):
        return float(int(pct))
    return pct


def format_coverage_impact_result(
    *,
    associated: int,
    total: int,
    pct: float,
    signature: str,
) -> str:
    return (
        f"~{pct:g}% of failing patterns ({associated}/{total}) associated with "
        f"this constraint signature ({signature}) — estimate only"
    )


def format_overall_coverage_impact(
    *,
    associated: int,
    total: int,
    pct: float,
) -> str:
    return (
        f"~{pct:g}% of all failing patterns ({associated}/{total}) are associated with "
        f"ATPG constraints (Reset / Scan Enable / Clock) across the dataset — estimate only"
    )


def format_category_coverage_impact(
    *,
    category_label: str,
    associated: int,
    total: int,
    pct: float,
) -> str:
    return (
        f"~{pct:g}% of all failing patterns ({associated}/{total}) are associated with "
        f"{category_label} constraints across the dataset — estimate only"
    )


def _build_summary_and_rows(
    cases: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    all_failing = _all_failing_pattern_labels(cases)
    total = len(all_failing) or 1

    violations = build_constraint_violation_results(cases)

    # Patterns touched by any ATPG constraint signature (deduped across categories).
    overall_patterns: Set[str] = set()
    # Per category union
    by_cat_patterns: Dict[str, Set[str]] = defaultdict(set)
    # Per unique fanout (avoid triple-count when expanding categories later for display)
    by_fanout_patterns: Dict[str, Set[str]] = defaultdict(set)

    rows: List[Dict[str, Any]] = []
    for v in violations:
        labels = [str(p) for p in (v.get("patternLabels") or [])]
        associated_set = set(labels) if labels else set()
        associated = len(associated_set) if associated_set else int(v.get("failingPatternCount") or 0)
        count_field = int(v.get("failingPatternCount") or 0)
        if count_field > associated:
            associated = min(count_field, total)
        if not associated_set and count_field > 0:
            # Synthetic placeholders only when labels missing — prefer real IDs
            associated_set = {f"P{i}" for i in range(1, associated + 1)}

        category = str(v.get("constraintCategory") or "")
        pin = str(v.get("heldPins") or "")
        fanout = str(v.get("fanoutSignal") or "unknown")
        signature = f"{pin} × {fanout}"

        overall_patterns.update(associated_set)
        if category:
            by_cat_patterns[category].update(associated_set)
        by_fanout_patterns[fanout].update(associated_set)

        pct_disp = _fmt_pct((associated / total) * 100)
        result = format_coverage_impact_result(
            associated=associated,
            total=total,
            pct=pct_disp,
            signature=signature,
        )
        rows.append(
            {
                "result": result,
                "constraintCategory": category,
                "constraintCategoryLabel": v.get("constraintCategoryLabel") or category,
                "heldPins": pin,
                "fanoutSignal": fanout,
                "signature": signature,
                "associatedPatterns": associated,
                "totalFailingPatterns": total,
                "coverageImpactPct": pct_disp,
                "patternLabels": labels,
                "failingPatternCount": count_field,
                "affectedDies": v.get("affectedDies") or 1,
                "lotId": v.get("lotId"),
                "dieLabel": v.get("dieLabel"),
                "procedure": v.get("procedure"),
                "estimateOnly": True,
                "scope": "signature",
            }
        )

    overall_n = len(overall_patterns)
    overall_pct = _fmt_pct((overall_n / total) * 100)
    category_summary: Dict[str, Dict[str, Any]] = {}
    for cat in _CATEGORY_ORDER:
        n = len(by_cat_patterns.get(cat, set()))
        pct = _fmt_pct((n / total) * 100)
        label = {"reset": "Reset", "scan_enable": "Scan Enable", "clock": "Clock"}.get(cat, cat)
        category_summary[cat] = {
            "constraintCategory": cat,
            "constraintCategoryLabel": label,
            "associatedPatterns": n,
            "totalFailingPatterns": total,
            "coverageImpactPct": pct,
            "result": format_category_coverage_impact(
                category_label=label,
                associated=n,
                total=total,
                pct=pct,
            ),
        }

    summary = {
        "overallAssociatedPatterns": overall_n,
        "totalFailingPatterns": total,
        "overallCoverageImpactPct": overall_pct,
        "overallResult": format_overall_coverage_impact(
            associated=overall_n,
            total=total,
            pct=overall_pct,
        ),
        "byCategory": category_summary,
        "uniqueFanouts": len(by_fanout_patterns),
    }

    # Put whole-data + category summaries first in diagnosis list for the UI.
    summary_rows: List[Dict[str, Any]] = [
        {
            "result": summary["overallResult"],
            "constraintCategory": "all",
            "constraintCategoryLabel": "All ATPG Constraints",
            "heldPins": "RESET_N / TEST / XI",
            "fanoutSignal": "all",
            "signature": "all ATPG constraints",
            "associatedPatterns": overall_n,
            "totalFailingPatterns": total,
            "coverageImpactPct": overall_pct,
            "patternLabels": sorted(overall_patterns)[:40],
            "affectedDies": len(
                {c.get("die_label") for c in cases if str(c.get("source_file", "")).startswith("fail")}
            ),
            "estimateOnly": True,
            "scope": "overall",
            "rank": 0,
        }
    ]
    for cat in _CATEGORY_ORDER:
        info = category_summary[cat]
        summary_rows.append(
            {
                **info,
                "heldPins": {
                    "reset": "RESET_N=1",
                    "scan_enable": "TEST=1",
                    "clock": "XI=0",
                }.get(cat, ""),
                "fanoutSignal": "all",
                "signature": f"{info['constraintCategoryLabel']} (all fan-outs)",
                "patternLabels": sorted(by_cat_patterns.get(cat, set()))[:40],
                "estimateOnly": True,
                "scope": "category",
                "rank": 0,
            }
        )

    rows.sort(
        key=lambda r: (
            -float(r.get("coverageImpactPct") or 0),
            _CATEGORY_ORDER.index(r["constraintCategory"])
            if r.get("constraintCategory") in _CATEGORY_ORDER
            else 99,
            str(r.get("fanoutSignal") or ""),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return summary, summary_rows + rows


def build_coverage_impact_results(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Overall + category summaries, then per-signature rows."""
    global _COVERAGE_ROWS_CACHE, _COVERAGE_SUMMARY_CACHE
    fingerprint = _inputs_fingerprint()
    if _COVERAGE_ROWS_CACHE is not None and _COVERAGE_ROWS_CACHE[0] == fingerprint:
        return _COVERAGE_ROWS_CACHE[1]

    persisted = _load_persisted(fingerprint)
    if persisted is not None:
        rows, summary = persisted
        _COVERAGE_ROWS_CACHE = (fingerprint, rows)
        _COVERAGE_SUMMARY_CACHE = (fingerprint, summary)
        return rows

    summary, rows = _build_summary_and_rows(cases)
    _persist(fingerprint, rows, summary)
    _COVERAGE_ROWS_CACHE = (fingerprint, rows)
    _COVERAGE_SUMMARY_CACHE = (fingerprint, summary)
    return rows


def get_coverage_impact_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    global _COVERAGE_SUMMARY_CACHE
    fingerprint = _inputs_fingerprint()
    if _COVERAGE_SUMMARY_CACHE is not None and _COVERAGE_SUMMARY_CACHE[0] == fingerprint:
        return _COVERAGE_SUMMARY_CACHE[1]
    build_coverage_impact_results(cases)
    if _COVERAGE_SUMMARY_CACHE is not None and _COVERAGE_SUMMARY_CACHE[0] == fingerprint:
        return _COVERAGE_SUMMARY_CACHE[1]
    return {}


def coverage_impact_kpi_value(cases: List[Dict[str, Any]]) -> str:
    """
    Card value: whole-dataset share of failing patterns tied to any ATPG constraint.
    """
    summary = get_coverage_impact_summary(cases)
    pct = float(summary.get("overallCoverageImpactPct") or 0)
    if pct >= 10 and pct == int(pct):
        return f"~{int(pct)}%"
    return f"~{pct:g}%"


def count_coverage_impact_rows(cases: List[Dict[str, Any]]) -> int:
    return len(build_coverage_impact_results(cases))


def _load_persisted(
    fingerprint: str,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    if not os.path.exists(COVERAGE_IMPACT_PATH):
        return None
    try:
        with open(COVERAGE_IMPACT_PATH, encoding="utf-8") as f:
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
    global _COVERAGE_SUMMARY_CACHE
    try:
        os.makedirs(os.path.dirname(COVERAGE_IMPACT_PATH), exist_ok=True)
        by_cat = Counter(
            r.get("constraintCategory") or "other"
            for r in rows
            if r.get("scope") == "signature"
        )
        with open(COVERAGE_IMPACT_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "artifact": "coverage_impact",
                    "fingerprint": fingerprint,
                    "count": len(rows),
                    "totalFailingPatterns": summary.get("totalFailingPatterns"),
                    "overallCoverageImpactPct": summary.get("overallCoverageImpactPct"),
                    "overallAssociatedPatterns": summary.get("overallAssociatedPatterns"),
                    "byCategory": by_cat,
                    "summary": summary,
                    "rows": rows,
                },
                f,
                indent=2,
            )
        _COVERAGE_SUMMARY_CACHE = (fingerprint, summary)
    except OSError:
        pass


def warm_coverage_impact_cache(cases: Optional[List[Dict[str, Any]]] = None) -> int:
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return 0
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if not isinstance(cases, list):
            return 0
    return count_coverage_impact_rows(cases)


def clear_coverage_impact_caches() -> None:
    global _COVERAGE_ROWS_CACHE, _COVERAGE_SUMMARY_CACHE
    _COVERAGE_ROWS_CACHE = None
    _COVERAGE_SUMMARY_CACHE = None
