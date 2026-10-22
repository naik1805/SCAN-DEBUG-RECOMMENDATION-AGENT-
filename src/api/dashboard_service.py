"""Build dashboard + KPI workspace payloads from compiled scan debug data."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.data.action_labels import (
    CATEGORY_COLORS,
    action_label_from_internal,
    category_from_internal,
)
from src.config import get_settings
from src.data.dataset_builder import build_compiled_dataset
from src.data.paths import COMPILED_DATASET_PATH, HISTORICAL_CASES_PATH
from src.data.chain_break_diagnosis import (
    HOW_TO_IMPLEMENT as BROKEN_CHAINS_HOW_TO_IMPLEMENT,
    build_broken_chain_diagnosis_results,
)
from src.data.atpg_constraint_violations import (
    build_constraint_violation_results,
    count_constraint_violations,
)
from src.data.atpg_constraint_review_recs import (
    build_constraint_review_recommendations,
    count_constraint_review_recommendations,
)
from src.data.atpg_coverage_impact import (
    build_coverage_impact_results,
    coverage_impact_kpi_value,
)
from src.data.timing_violations import (
    build_timing_violation_results,
    count_timing_violations,
)
from src.data.timing_debug_recs import (
    build_timing_debug_recommendations,
    count_timing_debug_recommendations,
)
from src.data.worst_slack import (
    build_worst_slack_results,
    get_worst_slack_summary,
    worst_slack_kpi_value,
)
from src.data.power_violations import (
    build_power_violation_results,
    build_power_violation_workspace_results,
    count_power_violations,
    get_power_violation_summary,
    power_violations_kpi_value,
)
from src.data.power_debug_recs import (
    build_power_debug_recommendations,
    build_power_debug_recs_workspace,
    count_power_debug_recommendations,
)
from src.data.peak_switching import (
    build_peak_switching_workspace,
    get_peak_switching_summary,
    peak_switching_kpi_value,
)
from src.data.defect_suspects import (
    build_defect_suspect_workspace,
    count_defect_suspects,
    get_defect_suspect_summary,
)
from src.data.investigation_recs import (
    build_investigation_recs_workspace,
    count_investigation_recommendations,
    get_investigation_recs_summary,
)
from src.data.defect_localization import (
    average_defect_localization_confidence,
    build_defect_localization_workspace,
    get_defect_localization_summary,
)
from src.data.scan_chain_confidence import (
    average_scan_chain_confidence,
    build_scan_chain_confidence_results,
    compute_scan_chain_confidence,
)
from src.data.scan_chain_debug_recs import (
    build_scan_chain_debug_recommendations,
    warm_scan_chain_debug_recs_cache,
)

CaseFilter = Callable[[Dict[str, Any]], bool]

from src.data.recommendation_engine import (
    ATPG_CONSTRAINT,
    PHYSICAL_DEFECT,
    POWER_DEBUG,
    SCAN_CHAIN,
    TIMING_DEBUG,
    build_recommendation_row,
    build_recommendations,
    build_root_cause_distribution,
    _case_confidence as engine_case_confidence,
    _priority_for_case,
)

SECTION_ACTION = {
    "scan_chain_debug": SCAN_CHAIN,
    "atpg_constraint_review": ATPG_CONSTRAINT,
    "timing_debug": TIMING_DEBUG,
    "power_related_debug": POWER_DEBUG,
    "physical_defect_investigation": PHYSICAL_DEFECT,
}


CATEGORY_DISPLAY = {
    "SCAN_CHAIN_DEBUG": "Broken Chain",
    "TIMING_DEBUG": "Timing",
    "POWER_RELATED_DEBUG": "Power",
    "ATPG_CONSTRAINT_REVIEW": "ATPG Constraint",
    "PHYSICAL_DEFECT_INVESTIGATION": "Physical Defect",
}

PRIORITY_DISPLAY = {"P0": "Critical", "P1": "High", "P2": "Medium", "P3": "Low"}


def _category_label(category: str) -> str:
    return CATEGORY_DISPLAY.get(category, category.replace("_", " ").title())


def _root_cause_for_case(case: Dict[str, Any]) -> str:
    from src.data.recommendation_engine import build_recommendation_row

    return build_recommendation_row(case, 0)["rootCause"]


def _scan_chain_id(case: Dict[str, Any]) -> str:
    if case.get("chain_name"):
        return str(case["chain_name"])
    if case.get("timing_chain"):
        return str(case["timing_chain"])
    die = case.get("die_label", "scan").replace("fail_die_", "")
    lot = case.get("lot_id", "LOT").replace("LOT_", "")
    return f"SC-{lot}{die.zfill(3)}"


def _expected_impact(case: Dict[str, Any], conf: float) -> str:
    action = case.get("true_action", "")
    if action == SCAN_CHAIN:
        return "Restore chain integrity"
    if action == TIMING_DEBUG:
        slack = case.get("min_slack", 9999)
        if slack < 9999:
            return f"{abs(int(slack))} ps slack recovery"
        return "Improve capture margin"
    if action == POWER_DEBUG:
        return "Stabilize capture power"
    if action == ATPG_CONSTRAINT:
        return f"+{round(0.4 + conf * 1.2, 1)}% coverage"
    if action == PHYSICAL_DEFECT:
        return f"+{round(0.5 + conf * 2, 1)}% yield"
    return f"+{round(0.4 + conf * 2, 1)}% yield"


def _cases_for_action(dataset: List[Dict[str, Any]], action: str) -> List[Dict[str, Any]]:
    return [c for c in dataset if c.get("true_action") == action]


def _is_critical_scan_chain_case(case: Dict[str, Any]) -> bool:
    """Highest-priority Inspect Scan Chain dies: routed action + break + strong mismatch."""
    return (
        case.get("true_action") == SCAN_CHAIN
        and case.get("has_break")
        and (case.get("mismatch_count") or 0) >= 40
    )


def _critical_scan_chains(dataset: List[Dict[str, Any]]) -> int:
    return sum(1 for c in dataset if _is_critical_scan_chain_case(c))


def _avg_conf_for_cases(cases: List[Dict[str, Any]]) -> str:
    if not cases:
        return "0%"
    vals = [
        min(0.99, 0.65 + c.get("cell_count", 1) * 0.01 + (0.1 if c["has_break"] else 0))
        for c in cases
    ]
    return f"{sum(vals) / len(vals) * 100:.0f}%"


def _coverage_impact_for_cases(cases: List[Dict[str, Any]]) -> str:
    if not cases:
        return "0%"
    return f"-{min(5.0, len(cases) * 0.35):.1f}%"


def _fail_die_cases(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in dataset if str(c.get("source_file", "")).startswith("fail")]


KPI_DEFS = [
    (
        "broken_chains",
        "scan_chain_debug",
        "Broken Chains Detected",
        lambda d: sum(1 for c in d if c.get("has_break")),
        5,
    ),
    (
        "debug_recommendations",
        "scan_chain_debug",
        "Scan Chain Debug Recommendations",
        _critical_scan_chains,
        10,
    ),
    (
        "avg_ai_confidence",
        "scan_chain_debug",
        "Scan Chain Confidence",
        lambda d: average_scan_chain_confidence(_cases_for_action(d, SCAN_CHAIN)),
        "90%",
    ),
    (
        "constraint_violations",
        "atpg_constraint_review",
        "Constraint Violations",
        lambda d: count_constraint_violations(_fail_die_cases(d)),
        5,
    ),
    (
        "pending_review",
        "atpg_constraint_review",
        "Review Recommendation",
        lambda d: count_constraint_review_recommendations(_fail_die_cases(d)),
        6,
    ),
    (
        "coverage_impact",
        "atpg_constraint_review",
        "Coverage Impact",
        lambda d: coverage_impact_kpi_value(_fail_die_cases(d)),
        "<1%",
    ),
    (
        "timing_violations",
        "timing_debug",
        "Timing Violations",
        lambda d: count_timing_violations(d),
        3,
    ),
    (
        "timing_debug_recs",
        "timing_debug",
        "Timing Debug Recommendations",
        lambda d: count_timing_debug_recommendations(d),
        4,
    ),
    (
        "worst_slack",
        "timing_debug",
        "Worst Slack",
        lambda d: worst_slack_kpi_value(d),
        ">0ps",
    ),
    (
        "power_violations",
        "power_related_debug",
        "Power Violations",
        lambda d: power_violations_kpi_value(d),
        2,
    ),
    (
        "power_debug_recs",
        "power_related_debug",
        "Power Debug Recommendations",
        lambda d: count_power_debug_recommendations(d),
        3,
    ),
    (
        "peak_switching",
        "power_related_debug",
        "Peak Switching",
        lambda d: peak_switching_kpi_value(d),
        "avg",
    ),
    (
        "defect_suspects",
        "physical_defect_investigation",
        "Defect Suspects",
        lambda d: count_defect_suspects(d),
        10,
    ),
    (
        "investigation_recs",
        "physical_defect_investigation",
        "Investigation Recommendations",
        lambda d: count_investigation_recommendations(d),
        5,
    ),
    (
        "defect_localization",
        "physical_defect_investigation",
        "Defect Localization",
        lambda d: average_defect_localization_confidence(d),
        "95%",
    ),
]

KPI_FILTERS: Dict[str, CaseFilter] = {
    "broken_chains": lambda c: bool(c.get("has_break")),
    "debug_recommendations": lambda c: bool(c.get("has_break")),
    "avg_ai_confidence": lambda c: bool(c.get("has_break")),
    "constraint_violations": lambda c: bool(
        str(c.get("source_file", "")).startswith("fail")
    ),
    "pending_review": lambda c: bool(
        str(c.get("source_file", "")).startswith("fail")
    ),
    "coverage_impact": lambda c: bool(
        str(c.get("source_file", "")).startswith("fail")
    ),
    "timing_violations": lambda c: (
        c.get("true_action") == TIMING_DEBUG
        and float(c.get("min_slack", 9999) or 9999) < 9999
    ),
    "timing_debug_recs": lambda c: c["true_action"] == TIMING_DEBUG,
    "worst_slack": lambda c: c["true_action"] == TIMING_DEBUG and c["min_slack"] < 9999,
    "power_violations": lambda c: (
        bool(str(c.get("source_file", "")).startswith("fail"))
        and (
            c.get("true_action") == POWER_DEBUG
            or str(c.get("defect_type") or "").upper() in ("CENTER", "NEAR_FULL")
        )
    ),    "power_debug_recs": lambda c: c["true_action"] == POWER_DEBUG,
    "peak_switching": lambda c: c["true_action"] == POWER_DEBUG,
    "defect_suspects": lambda c: c["true_action"] == PHYSICAL_DEFECT,
    "investigation_recs": lambda c: c["true_action"] == PHYSICAL_DEFECT,
    "defect_localization": lambda c: c["true_action"] == PHYSICAL_DEFECT,
}

KPI_TOOLTIPS: Dict[str, str] = {
    "broken_chains": "Broken scan chains — inspect critical chains and locate shifter failures",
    "debug_recommendations": "Most critical dies recommended for Inspect Scan Chain (break + high mismatch)",
    "avg_ai_confidence": "Weighted confidence: pattern consistency × (1/ambiguity) × historical match boost",
    "constraint_violations": "Typed ATPG over-constraints: Reset / Scan Enable / Clock (held pin × fan-out)",
    "pending_review": "Category-specific ATPG review (Reset / Scan Enable / Clock) + historical resolution cite",
    "coverage_impact": "Whole-dataset share of failing patterns tied to any ATPG constraint (Reset/SE/Clock); estimate only",
    "timing_violations": "At-speed/timing-correlated pattern fails (fail only above freq threshold) + STIL WaveformTable edge spacing",
    "timing_debug_recs": "Review capture clock timing per pattern/chain/domain + historical frequency cites",
    "worst_slack": "Frequency margin proxy (fail vs pass MHz) + worst slack in ps when available",
    "power_violations": "Patterns with IR_DROP_MV or THERMAL_C above threshold (flagged even if STATUS=PASS)",
    "power_debug_recs": "Check IR-drop during capture per pattern (measured vs threshold) + historical IR-drop fail cites",
    "peak_switching": "MAX(IR_DROP_MV) across patterns as switching-activity proxy vs run average",
    "defect_suspects": "Top-N diagnosis candidates validated by failing-pattern consistency + STIL net/cell resolution",
    "investigation_recs": "Investigate top defect-suspect nets after TF/IR cross-check + historical PFA cite",
    "defect_localization": "Average localization confidence % from suspects + TF/IR + FR-009 XY + historical PFA",
}

KPI_VIZ_TYPES: Dict[str, str] = {
    "broken_chains": "topology",
    "debug_recommendations": "priority_matrix",
    "avg_ai_confidence": "gauge",
    "constraint_violations": "dependency",
    "pending_review": "assignment",
    "coverage_impact": "heatmap",
    "timing_violations": "timing",
    "timing_debug_recs": "clock_tree",
    "worst_slack": "critical_path",
    "power_violations": "power",
    "power_debug_recs": "power",
    "peak_switching": "heatmap",
    "defect_suspects": "wafer",
    "investigation_recs": "history",
    "defect_localization": "wafer",
}

KPI_PRIMARY_ACTION: Dict[str, str] = {
    "broken_chains": "INSPECT_SCAN_CHAIN",
    "debug_recommendations": "INSPECT_SCAN_CHAIN",
    "avg_ai_confidence": "INSPECT_SCAN_CHAIN",
    "constraint_violations": "REVIEW_ATPG_CONSTRAINTS",
    "pending_review": "REVIEW_ATPG_CONSTRAINTS",
    "coverage_impact": "REVIEW_ATPG_CONSTRAINTS",
    "timing_violations": "REVIEW_CAPTURE_CLOCK_TIMING",
    "timing_debug_recs": "REVIEW_CAPTURE_CLOCK_TIMING",
    "worst_slack": "REVIEW_CAPTURE_CLOCK_TIMING",
    "power_violations": "CHECK_IR_DROP_DURING_CAPTURE",
    "power_debug_recs": "CHECK_IR_DROP_DURING_CAPTURE",
    "peak_switching": "CHECK_IR_DROP_DURING_CAPTURE",
    "defect_suspects": "INVESTIGATE_PHYSICAL_DEFECT",
    "investigation_recs": "INVESTIGATE_PHYSICAL_DEFECT",
    "defect_localization": "INVESTIGATE_PHYSICAL_DEFECT",
}

KPI_ROOT_CAUSE: Dict[str, str] = {
    "broken_chains": "Broken scan chains detected — inspect critical chains and trace shifter failures before pattern release.",
    "debug_recommendations": "Scan chain debug path selected — inspect affected chains and validate continuity on failing dies.",
    "avg_ai_confidence": "Confidence spread across scan chain debug recommendations in the active dataset.",
    "constraint_violations": "ATPG constraint or mask mismatch reducing effective scan coverage on affected patterns.",
    "pending_review": "ATPG constraint review recommendations awaiting engineer sign-off before pattern release.",
    "coverage_impact": "Failing-pattern share per constraint signature (bitmap proxy; estimate only).",
    "timing_violations": "Pattern fails only at high-frequency timing set; STIL capture edge spacing near minimum margin.",
    "timing_debug_recs": "Review capture clock timing for the pattern capture window; cite historical frequency fails.",
    "worst_slack": "Fails at high MHz, passes at slower MHz — frequency margin proxy; include worst slack ps when known.",
    "power_violations": "IR drop or thermal above threshold on a pattern — flagged even when STATUS=PASS.",
    "power_debug_recs": "Check IR-drop during capture for Pattern #N (XmV, Y% above threshold); cite similar historical IR fails within ±5 patterns.",
    "peak_switching": "Peak IR-drop: XmV at Pattern #N (vs. avg YmV across run).",
    "defect_suspects": "Net N#### (Ua→Ub) — diagnosis rank 1, consistent with C/T failing patterns; …",
    "investigation_recs": "Investigate Net N#### (Ua→Ub) — suspected bridging; TF/IR rules out power false-fail; historical PFA.",
    "defect_localization": "Defect localization confidence: NN% — Net N#### at wafer (x,y), rank, consistency, PFA precedent.",
}


def _load_dataset() -> List[Dict[str, Any]]:
    if not os.path.exists(COMPILED_DATASET_PATH):
        build_compiled_dataset(write=True)
    with open(COMPILED_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _avg_conf(dataset: List[Dict[str, Any]]) -> str:
    if not dataset:
        return "0%"
    vals = [min(0.99, 0.65 + c.get("cell_count", 1) * 0.01 + (0.1 if c["has_break"] else 0)) for c in dataset]
    return f"{sum(vals) / len(vals) * 100:.0f}%"


def _worst_slack(dataset: List[Dict[str, Any]]) -> str:
    slacks = [c["min_slack"] for c in dataset if c["min_slack"] < 9999]
    if not slacks:
        return "N/A"
    return f"{min(slacks):.0f}ps"


def _spark(seed: int) -> List[int]:
    return [max(2, int(seed + (i % 5) * 2)) for i in range(12)]


def _kpi_status(value: Any, target: Any) -> str:
    try:
        if isinstance(value, str) and "%" in value and isinstance(target, str) and "%" in target:
            return "improving" if float(value.replace("%", "")) >= float(target.replace("%", "")) else "at_risk"
        if isinstance(value, (int, float)) and isinstance(target, (int, float)):
            return "breach" if value > target else "on_track"
    except ValueError:
        pass
    return "at_risk"


def _die_chain_hint(case: Dict[str, Any]) -> str:
    label = case.get("die_label", "")
    match = re.search(r"(\d+)$", label)
    return match.group(1) if match else ""


def _action_display(case: Dict[str, Any], include_chain: bool = True) -> str:
    from src.data.recommendation_engine import build_recommendation_row

    row = build_recommendation_row(case, 0)
    if include_chain:
        return row["recommendation"]
    return action_label_from_internal(case.get("true_action", ""))


def _case_confidence(case: Dict[str, Any]) -> float:
    return engine_case_confidence(case)


def _filter_cases(dataset: List[Dict[str, Any]], kpi_id: str) -> List[Dict[str, Any]]:
    predicate = KPI_FILTERS.get(kpi_id, lambda _c: True)
    return [c for c in dataset if predicate(c)]


def _build_breakdown(cases: List[Dict[str, Any]], kpi_id: str) -> List[Dict[str, Any]]:
    total = max(len(cases), 1)
    breakdown: List[Dict[str, Any]] = []

    if kpi_id in ("timing_violations", "timing_debug_recs", "worst_slack"):
        buckets = Counter(
            "Critical (<0ps)" if c["min_slack"] < 0 else "Violation (<9999ps)" for c in cases
        )
        for name, count in buckets.most_common(4):
            breakdown.append({"dimension": "Slack Bucket", "value": name, "share": round(count * 100 / total)})
        for lot, count in Counter(c["lot_id"] for c in cases).most_common(3):
            breakdown.append({"dimension": "Lot", "value": lot, "share": round(count * 100 / total)})
    elif kpi_id in ("power_violations", "power_debug_recs", "peak_switching"):
        for defect, count in Counter(c.get("defect_type", "NORMAL") for c in cases).most_common(4):
            breakdown.append({"dimension": "Defect Shape", "value": defect, "share": round(count * 100 / total)})
        for lot, count in Counter(c["lot_id"] for c in cases).most_common(3):
            breakdown.append({"dimension": "Lot", "value": lot, "share": round(count * 100 / total)})
    elif kpi_id in ("defect_suspects", "investigation_recs", "defect_localization"):
        for defect, count in Counter(c.get("defect_type", "NORMAL") for c in cases).most_common(4):
            breakdown.append({"dimension": "Bitmap Pattern", "value": defect, "share": round(count * 100 / total)})
        density = Counter("High" if c["cell_count"] >= 5 else "Medium" if c["cell_count"] >= 2 else "Low" for c in cases)
        for band, count in density.most_common(3):
            breakdown.append({"dimension": "Cell Density", "value": band, "share": round(count * 100 / total)})
    elif kpi_id in ("constraint_violations", "pending_review", "coverage_impact"):
        for lot, count in Counter(c["lot_id"] for c in cases).most_common(4):
            breakdown.append({"dimension": "Lot", "value": lot, "share": round(count * 100 / total)})
        for action, count in Counter(c.get("true_action", "") for c in cases).most_common(3):
            breakdown.append(
                {
                    "dimension": "Action",
                    "value": action_label_from_internal(action),
                    "share": round(count * 100 / total),
                }
            )
    else:
        lot_counts = Counter(c["lot_id"] for c in cases)
        for lot, count in lot_counts.most_common(4):
            breakdown.append({"dimension": "Lot", "value": lot, "share": round(count * 100 / total)})
        if kpi_id == "broken_chains":
            critical = Counter(
                "Critical" if c["mismatch_count"] >= 40 else "Monitor"
                for c in cases
            )
            for band, count in critical.most_common(2):
                breakdown.append({"dimension": "Chain Priority", "value": band, "share": round(count * 100 / total)})
        for die, count in Counter(c.get("die_label", "") for c in cases).most_common(3):
            breakdown.append(
                {
                    "dimension": "Die to Inspect",
                    "value": die,
                    "share": round(count * 100 / total),
                }
            )

    return breakdown[:9]


def _build_viz_series(cases: List[Dict[str, Any]], kpi_id: str) -> List[Dict[str, Any]]:
    if not cases:
        return [{"label": "No cases", "value": 0}]

    if kpi_id == "avg_ai_confidence":
        avg = sum(compute_scan_chain_confidence(c)["confidenceScore"] for c in cases) / len(cases)
        return [{"label": "Mean confidence", "value": round(avg * 100)}]

    if kpi_id == "constraint_violations":
        rows = build_constraint_violation_results(cases)
        cat_counts: Counter[str] = Counter()
        for row in rows:
            label = str(row.get("constraintCategoryLabel") or row.get("constraintCategory") or "Other")
            cat_counts[label] += 1
        return [{"label": k, "value": v} for k, v in cat_counts.most_common()] or [
            {"label": "No violations", "value": 0}
        ]

    if kpi_id == "pending_review":
        rows = build_constraint_review_recommendations(cases)
        cat_counts: Counter[str] = Counter()
        for row in rows:
            label = str(row.get("constraintCategoryLabel") or row.get("constraintCategory") or "Other")
            cat_counts[label] += 1
        return [{"label": k, "value": v} for k, v in cat_counts.most_common()] or [
            {"label": "No recommendations", "value": 0}
        ]

    if kpi_id == "coverage_impact":
        rows = build_coverage_impact_results(cases)
        return [
            {
                "label": str(r.get("signature") or r.get("fanoutSignal") or "sig"),
                "value": float(r.get("coverageImpactPct") or 0),
            }
            for r in rows[:8]
        ] or [{"label": "No impact", "value": 0}]

    if kpi_id == "timing_violations":
        rows = build_timing_violation_results(cases)
        kind_counts: Counter[str] = Counter()
        for row in rows:
            kind_counts[str(row.get("kind") or "timing").title()] += 1
        return [{"label": k, "value": v} for k, v in kind_counts.most_common()] or [
            {"label": "No timing violations", "value": 0}
        ]

    if kpi_id == "timing_debug_recs":
        rows = build_timing_debug_recommendations(cases)
        kind_counts: Counter[str] = Counter()
        for row in rows:
            kind_counts[str(row.get("kind") or "timing").title()] += 1
        return [{"label": k, "value": v} for k, v in kind_counts.most_common()] or [
            {"label": "No recommendations", "value": 0}
        ]

    if kpi_id == "worst_slack":
        rows = build_worst_slack_results(cases)
        return [
            {
                "label": str(r.get("patternLabel") or r.get("dieLabel") or "?"),
                "value": abs(int(float(r.get("worstSlackPs") or 0))),
            }
            for r in rows[:8]
        ] or [{"label": "No slack", "value": 0}]

    if kpi_id == "power_violations":
        rows = build_power_violation_workspace_results(cases)
        kind_counts: Counter[str] = Counter()
        for row in rows:
            kind = str(row.get("kind") or "other")
            label = {"ir_drop": "IR Drop", "thermal": "Thermal", "both": "IR+Thermal"}.get(kind, kind)
            kind_counts[label] += 1
        return [{"label": k, "value": v} for k, v in kind_counts.most_common()] or [
            {"label": "No violations", "value": 0}
        ]

    if kpi_id == "power_debug_recs":
        rows = build_power_debug_recs_workspace(cases)
        kind_counts: Counter[str] = Counter()
        for row in rows:
            kind = str(row.get("kind") or "other")
            label = {"ir_drop": "IR Drop", "thermal": "Thermal", "both": "IR+Thermal"}.get(kind, kind)
            kind_counts[label] += 1
        return [{"label": k, "value": v} for k, v in kind_counts.most_common()] or [
            {"label": "No recommendations", "value": 0}
        ]

    if kpi_id == "peak_switching":
        rows = build_peak_switching_workspace(cases)
        return [
            {
                "label": str(r.get("patternLabel") or r.get("patternId") or "?"),
                "value": float(r.get("irDropMv") or 0),
            }
            for r in rows[:8]
        ] or [{"label": "No IR data", "value": 0}]

    if kpi_id == "defect_suspects":
        rows = build_defect_suspect_workspace(cases)
        return [
            {
                "label": str(r.get("netId") or r.get("cellName") or "?"),
                "value": float(r.get("consistencyRatio") or 0) * 100.0,
            }
            for r in rows[:8]
        ] or [{"label": "No suspects", "value": 0}]

    if kpi_id == "investigation_recs":
        rows = build_investigation_recs_workspace(cases)
        hyp_counts: Counter[str] = Counter()
        for row in rows:
            hyp_counts[str(row.get("faultHypothesis") or "other")] += 1
        return [{"label": k, "value": v} for k, v in hyp_counts.most_common()] or [
            {"label": "No recommendations", "value": 0}
        ]

    if kpi_id == "defect_localization":
        rows = build_defect_localization_workspace(cases)
        return [
            {
                "label": str(r.get("netId") or "?"),
                "value": float(r.get("confidencePct") or 0),
            }
            for r in rows[:8]
        ] or [{"label": "No localization", "value": 0}]

    ordered = sorted(cases, key=lambda c: c["mismatch_count"], reverse=True)[:6]
    if kpi_id == "broken_chains":
        series = []
        for c in ordered:
            label = c.get("chain_display") or c.get("chain_name") or c["die_label"]
            bit = c.get("candidate_bit")
            try:
                value = int(str(bit))
            except (TypeError, ValueError):
                value = c["mismatch_count"]
            series.append({"label": str(label), "value": value})
        return series or [{"label": "No cases", "value": 0}]
    return [{"label": f"{c['lot_id']}/{c['die_label']}", "value": c["mismatch_count"]} for c in ordered]


def _build_impact(kpi_id: str, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n = max(len(cases), 1)
    if kpi_id in ("timing_violations", "timing_debug_recs", "worst_slack"):
        return [
            {"label": "Worst Slack", "before": "−42ps", "after": "+8ps", "delta": "+50ps"},
            {"label": "Capture Margin", "before": "Failed", "after": "Pass", "delta": "Restored"},
            {"label": "Mismatch Count", "before": f"{sum(c['mismatch_count'] for c in cases)}", "after": f"{max(1, sum(c['mismatch_count'] for c in cases)//3)}", "delta": "−67%"},
        ]
    if kpi_id in ("power_violations", "power_debug_recs", "peak_switching"):
        return [
            {"label": "IR Drop", "before": "1.62×", "after": "1.12×", "delta": "−31%"},
            {"label": "Peak Switching", "before": "1.8×", "after": "1.3×", "delta": "−28%"},
            {"label": "Capture Pass Rate", "before": f"{max(1, 100 - n)}%", "after": "96%", "delta": f"+{min(12, n)}pp"},
        ]
    if kpi_id in ("defect_suspects", "investigation_recs", "defect_localization"):
        return [
            {"label": "Localization", "before": "Unknown", "after": "Wafer XY", "delta": "Resolved"},
            {"label": "Suspect Cells", "before": str(sum(c["cell_count"] for c in cases)), "after": str(max(1, sum(c["cell_count"] for c in cases)//2)), "delta": "−50%"},
            {"label": "PFA Readiness", "before": "Low", "after": "High", "delta": "Ready"},
        ]
    if kpi_id in ("constraint_violations", "pending_review", "coverage_impact"):
        return [
            {"label": "Coverage", "before": "94.1%", "after": "97.8%", "delta": "+3.7pp"},
            {"label": "Constraint Hits", "before": str(n), "after": str(max(0, n - 2)), "delta": f"−{min(2, n)}"},
            {"label": "Pattern Readiness", "before": "Blocked", "after": "Release", "delta": "Unblocked"},
        ]
    return [
        {"label": "Broken Chains", "before": str(n), "after": str(max(0, n - 3)), "delta": f"−{min(3, n)}"},
        {"label": "Yield", "before": "91.1%", "after": "92.4%", "delta": "+1.3pp"},
        {"label": "Debug Cycle", "before": "5.1h", "after": "3.8h", "delta": "−25%"},
    ]


def _build_summary_cards(kpi: Dict[str, Any], cases: List[Dict[str, Any]], dataset: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cards = [
        {"label": "Current", "value": str(kpi["value"])},
        {"label": "Target", "value": str(kpi["target"])},
        {"label": "Matching Cases", "value": str(len(cases))},
        {"label": "Lots", "value": str(len({c["lot_id"] for c in cases}))},
    ]
    kpi_id = kpi["id"]

    if kpi_id in ("broken_chains", "debug_recommendations"):
        cards.append({"label": "Broken Chains", "value": str(sum(1 for c in dataset if c.get("has_break")))})
        cards.append({"label": "Critical to Inspect", "value": str(_critical_scan_chains(dataset))})
        cards.append({"label": "Recommended Action", "value": "Inspect Scan Chain"})
    elif kpi_id == "avg_ai_confidence":
        cards.append({"label": "Formula", "value": "pattern × 1/ambiguity × hist"})
        cards.append({"label": "High-Conf Cases", "value": str(sum(1 for c in cases if compute_scan_chain_confidence(c)["confidencePct"] >= 70))})
    elif kpi_id in ("timing_violations", "timing_debug_recs", "worst_slack"):
        cards.append({"label": "Worst Slack", "value": worst_slack_kpi_value(cases) if cases else "N/A"})
        cards.append({"label": "Primary Action", "value": "Review Capture Clock Timing"})
    elif kpi_id in ("power_violations", "power_debug_recs", "peak_switching"):
        cards.append({"label": "IR-Drop Cases", "value": str(len(cases))})
        cards.append({"label": "Primary Action", "value": "Check IR-Drop During Capture"})
    elif kpi_id in ("defect_suspects", "investigation_recs", "defect_localization"):
        cards.append({"label": "Suspect Cells", "value": str(sum(c["cell_count"] for c in cases))})
        cards.append({"label": "Primary Action", "value": "Investigate Physical Defect"})
    else:
        cards.append({"label": "Primary Action", "value": "Review ATPG Constraints"})

    cards.append({"label": "Dataset Total", "value": str(len(dataset))})
    return cards


_CATEGORY_TO_INTERNAL = {
    "SCAN_CHAIN_DEBUG": "INSPECT_SCAN_CHAIN",
    "TIMING_DEBUG": "REVIEW_CAPTURE_CLOCK_TIMING",
    "POWER_RELATED_DEBUG": "CHECK_IR_DROP_DURING_CAPTURE",
    "ATPG_CONSTRAINT_REVIEW": "REVIEW_ATPG_CONSTRAINTS",
    "PHYSICAL_DEFECT_INVESTIGATION": "INVESTIGATE_PHYSICAL_DEFECT",
}


def _kpi_recommendation_volumes(dataset: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Live recommendation volumes from the KPI engines (same sources as the KPI cards).
    Used for Failure Root Cause Distribution / Priority / Trend.
    """
    fail = _fail_die_cases(dataset)
    scan_recs = len(build_scan_chain_debug_recommendations(dataset))
    if scan_recs <= 0:
        scan_recs = sum(1 for c in dataset if c.get("has_break"))
    return {
        "SCAN_CHAIN_DEBUG": scan_recs,
        "TIMING_DEBUG": count_timing_debug_recommendations(dataset),
        "POWER_RELATED_DEBUG": count_power_debug_recommendations(dataset),
        "ATPG_CONSTRAINT_REVIEW": count_constraint_review_recommendations(fail),
        "PHYSICAL_DEFECT_INVESTIGATION": count_investigation_recommendations(dataset),
    }


def _build_root_cause_from_kpis(
    dataset: List[Dict[str, Any]],
    volumes: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Donut slices = recommendation counts per debug action (from KPI cards)."""
    volumes = volumes if volumes is not None else _kpi_recommendation_volumes(dataset)
    label_by_cat = {
        "SCAN_CHAIN_DEBUG": "Inspect Scan Chain",
        "TIMING_DEBUG": "Review Capture Clock Timing",
        "POWER_RELATED_DEBUG": "Check IR-Drop During Capture",
        "ATPG_CONSTRAINT_REVIEW": "Review ATPG Constraints",
        "PHYSICAL_DEFECT_INVESTIGATION": "Investigate Physical Defect",
    }
    out: List[Dict[str, Any]] = []
    for cat, count in sorted(volumes.items(), key=lambda kv: -kv[1]):
        if count <= 0:
            continue
        out.append(
            {
                "name": label_by_cat.get(cat, cat.replace("_", " ").title()),
                "value": int(count),
                "fill": CATEGORY_COLORS.get(cat, "#7C3AED"),
            }
        )
    if out:
        return out
    # Fallback to die true_action distribution if engines are empty
    return build_root_cause_distribution(dataset)


def _build_priority_from_kpis(
    dataset: List[Dict[str, Any]],
    volumes: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Priority bars from die-level case priority plus KPI recommendation volumes
    mapped by category severity (Critical / High / Medium / Low).
    """
    fail = _fail_die_cases(dataset)
    counts: Counter[str] = Counter()
    for case in fail:
        counts[_priority_for_case(case)[1]] += 1

    volumes = volumes if volumes is not None else _kpi_recommendation_volumes(dataset)
    # Map KPI recommendation volumes into priority tiers (product severity).
    # Die priorities already cover action assignment; add excess engine findings
    # so charts reflect the full KPI card landscape.
    die_by_action = Counter(
        category_from_internal(c.get("true_action", "")) for c in fail
    )
    # Critical ← broken-chain / scan-chain findings beyond die count
    counts["Critical"] += max(0, volumes["SCAN_CHAIN_DEBUG"] - die_by_action.get("SCAN_CHAIN_DEBUG", 0))
    # High ← timing + physical investigation recs beyond die counts
    counts["High"] += max(0, volumes["TIMING_DEBUG"] - die_by_action.get("TIMING_DEBUG", 0))
    counts["High"] += max(
        0, volumes["PHYSICAL_DEFECT_INVESTIGATION"] - die_by_action.get("PHYSICAL_DEFECT_INVESTIGATION", 0)
    )
    # Medium ← ATPG review + power debug recs beyond die counts
    counts["Medium"] += max(
        0, volumes["ATPG_CONSTRAINT_REVIEW"] - die_by_action.get("ATPG_CONSTRAINT_REVIEW", 0)
    )
    counts["Medium"] += max(
        0, volumes["POWER_RELATED_DEBUG"] - die_by_action.get("POWER_RELATED_DEBUG", 0)
    )
    # Low ← residual soft signals (coverage / localization present)
    low_extra = 0
    try:
        from src.data.defect_localization import get_defect_localization_summary

        loc = get_defect_localization_summary(dataset)
        if int(loc.get("count") or 0) > 0:
            low_extra += 1
    except Exception:
        pass
    counts["Low"] += low_extra

    order = [
        ("Critical", "#EF4444"),
        ("High", "#F59E0B"),
        ("Medium", "#7C3AED"),
        ("Low", "#64748B"),
    ]
    return [
        {"name": name, "value": int(counts.get(name, 0)), "fill": fill}
        for name, fill in order
        if int(counts.get(name, 0)) > 0
    ] or [{"name": "Medium", "value": 0, "fill": "#7C3AED"}]


def _build_recommendation_trend_from_kpis(
    dataset: List[Dict[str, Any]],
    volumes: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    30-day recommendation activity trend scaled to live KPI recommendation totals.
    Prefer historical created_at buckets when dates span multiple days; else
    distribute total KPI volume across the window with a stable weekly wave.
    """
    volumes = volumes if volumes is not None else _kpi_recommendation_volumes(dataset)
    total = max(1, sum(volumes.values()))
    today = datetime.utcnow().date()

    # Try historical case dates first
    by_day: Counter[str] = Counter()
    if os.path.exists(HISTORICAL_CASES_PATH):
        try:
            with open(HISTORICAL_CASES_PATH, encoding="utf-8") as f:
                hist = json.load(f)
            for case in hist.get("cases") or []:
                raw = case.get("created_at") or case.get("reviewed_at")
                if not raw:
                    continue
                day = str(raw)[:10]
                by_day[day] += 1
        except (OSError, json.JSONDecodeError, TypeError):
            by_day = Counter()

    if len(by_day) >= 5:
        # Scale historical daily counts so the window sums ≈ total KPI recs
        hist_sum = sum(by_day.values()) or 1
        scale = total / hist_sum
        out: List[Dict[str, Any]] = []
        for i in range(30):
            d = today - timedelta(days=29 - i)
            key = d.isoformat()
            raw_n = by_day.get(key, 0)
            # Fill empty days with a small baseline from weekly pattern
            if raw_n <= 0:
                raw_n = max(1, int((total / 30) * (0.4 + 0.2 * ((i % 7) / 6))))
            else:
                raw_n = max(1, int(round(raw_n * scale)))
            out.append({"date": d.strftime("%m-%d"), "value": raw_n})
        return out

    # Synthetic but KPI-scaled wave (stable, not random)
    base = total / 30.0
    out = []
    for i in range(30):
        # Weekly sawtooth similar to prior UI, amplitude from KPI volume
        wave = 0.55 + 0.45 * (((i % 7) + 1) / 7.0)
        # Mid-month / end-month peaks
        if i in (4, 11, 18, 25):
            wave *= 1.35
        value = max(1, int(round(base * wave)))
        d = today - timedelta(days=29 - i)
        out.append({"date": d.strftime("%m-%d"), "value": value})
    return out


_DASHBOARD_CACHE: Optional[Tuple[float, Dict[str, Any]]] = None


def build_dashboard_payload(agent_confidence: float = 0.87) -> Dict[str, Any]:
    global _DASHBOARD_CACHE
    now = datetime.utcnow().timestamp()
    ttl = get_settings().dashboard_cache_ttl_sec
    if _DASHBOARD_CACHE is not None and (now - _DASHBOARD_CACHE[0]) < ttl:
        cached = dict(_DASHBOARD_CACHE[1])
        cached["aiConfidence"] = agent_confidence
        return cached

    dataset = _load_dataset()
    hist = {}
    if os.path.exists(HISTORICAL_CASES_PATH):
        with open(HISTORICAL_CASES_PATH, "r", encoding="utf-8") as f:
            hist = json.load(f)

    volumes = _kpi_recommendation_volumes(dataset)
    root_cause = _build_root_cause_from_kpis(dataset, volumes=volumes)
    priority_distribution = _build_priority_from_kpis(dataset, volumes=volumes)
    trend = _build_recommendation_trend_from_kpis(dataset, volumes=volumes)
    approval_trend = [
        {"date": f"W{i+1}", "value": 0, "approved": 3 + i % 4, "rejected": 1 + i % 2, "pending": 2 + i % 3}
        for i in range(12)
    ]

    kpis = []
    for kpi_id, section, title, calc, target in KPI_DEFS:
        value = calc(dataset)
        if isinstance(value, float):
            display = f"{value:.1f}"
        else:
            display = value
        kpis.append(
            {
                "id": kpi_id,
                "section": section,
                "title": title,
                "value": display,
                "target": target,
                "trendPct": 3 if isinstance(value, (int, float)) and value else 0,
                "sparkline": _spark(hash(kpi_id) % 20 + 10),
                "status": _kpi_status(value, target),
                "severity": "critical" if kpi_id in ("broken_chains", "worst_slack", "timing_violations") else "high",
                "tooltip": KPI_TOOLTIPS.get(kpi_id, f"{title} from live scan debug dataset"),
            }
        )

    recommendations = build_recommendations(
        dataset, limit=get_settings().top_recommendations_limit
    )

    top = max(dataset, key=lambda c: c["mismatch_count"], default={})
    executive_summary = [
        {
            "id": "broken_chains",
            "label": "Broken Chains",
            "value": str(sum(1 for c in dataset if c.get("has_break"))),
            "detail": "All diagnosed scan chain breaks",
            "tone": "danger",
        },
        {
            "id": "timing_debug_recs",
            "label": "Timing Issues",
            "value": str(count_timing_debug_recommendations(dataset)),
            "detail": "Review capture clock timing recommendations",
            "tone": "warning",
        },
        {
            "id": "power_debug_recs",
            "label": "Power Issues",
            "value": str(count_power_debug_recommendations(dataset)),
            "detail": "Check IR-Drop During Capture recommendations",
            "tone": "warning",
        },
        {
            "id": "constraint_violations",
            "label": "Constraint Violations",
            "value": str(count_constraint_violations(_fail_die_cases(dataset))),
            "detail": "Review ATPG Constraints recommendations",
            "tone": "info",
        },
        {
            "id": "investigation_recs",
            "label": "Physical Defects",
            "value": str(count_investigation_recommendations(dataset)),
            "detail": "Investigate Physical Defect recommendations",
            "tone": "primary",
        },
        {
            "id": "coverage_impact",
            "label": "Coverage Impact",
            "value": coverage_impact_kpi_value(_fail_die_cases(dataset)),
            "detail": "Top constraint-signature share of failing patterns (estimate)",
            "tone": "info",
        },
        {
            "id": "debug_recommendations",
            "label": "Estimated Yield Improvement",
            "value": f"+{len(dataset) * 0.05:.1f}%",
            "detail": "Across active failing lots",
            "tone": "success",
        },
        {
            "id": "debug_time_saved",
            "label": "Expected Debug Time Reduction",
            "value": f"{round(len(dataset) * 0.4, 1)} hrs",
            "detail": "Estimated debug hours saved",
            "tone": "success",
        },
        {
            "id": "avg_ai_confidence",
            "label": "AI Confidence",
            "value": average_scan_chain_confidence(_cases_for_action(dataset, SCAN_CHAIN)),
            "detail": "Weighted pattern × ambiguity × historical boost",
            "tone": "info",
        },
    ]

    workflow = [
        {"id": "logs", "label": "Failure Logs", "status": "done"},
        {"id": "diag", "label": "Diagnosis Engine", "status": "done"},
        {"id": "rca", "label": "Root Cause Analysis", "status": "done"},
        {"id": "agent", "label": "Scan Debug Recommendation Agent", "status": "active"},
        {"id": "impl", "label": "Implementation", "status": "upcoming"},
        {"id": "val", "label": "Validation", "status": "upcoming"},
    ]

    payload = {
        "kpis": kpis,
        "rootCauseDistribution": root_cause,
        "recommendationPriority": priority_distribution,
        "recommendationTrend": trend,
        "aiConfidence": agent_confidence,
        "approvalTrend": approval_trend,
        "recommendations": recommendations,
        "executiveSummary": executive_summary,
        "workflow": workflow,
        "meta": {
            "dataset_cases": len(dataset),
            "historical_cases": hist.get("summary", {}).get("total_cases", 0),
            "recommendation_source": get_settings().recommendation_source,
        },
    }
    _DASHBOARD_CACHE = (now, payload)
    return payload


def build_kpi_workspace(kpi_id: str, agent_confidence: float = 0.87) -> Dict[str, Any]:
    dashboard = build_dashboard_payload(agent_confidence)
    kpi = next((k for k in dashboard["kpis"] if k["id"] == kpi_id), dashboard["kpis"][0])
    dataset = _load_dataset()
    cases = _filter_cases(dataset, kpi_id)

    primary_internal = KPI_PRIMARY_ACTION.get(kpi_id, SCAN_CHAIN)
    primary_label = action_label_from_internal(primary_internal)
    root_cause = KPI_ROOT_CAUSE.get(kpi_id, KPI_ROOT_CAUSE["debug_recommendations"])

    if not cases:
        root_cause = f"No active cases for this recommendation type. {root_cause}"

    decision = {
        "executiveSummary": (
            f"{kpi['title']} is {kpi['status'].replace('_', ' ')} versus target {kpi['target']}. "
            f"{len(cases)} matching case(s) in this drill-down."
        ),
        "rootCause": root_cause,
        "confidence": agent_confidence,
        "businessImpact": f"Covers {len(cases)} case(s) across {len({c['lot_id'] for c in cases})} lot(s).",
        "risk": kpi["severity"],
        "recommendation": primary_label,
        "whatFailed": f"{kpi['title']} at {kpi['value']} (target {kpi['target']}).",
        "whyAiRecommended": (
            build_recommendation_row(cases[0], 1)["evidence"]
            if cases
            else (
                f"Evidence-driven recommendation: {primary_label} selected from diagnosis chain breaks, "
                "capture slack, defect signature, and historical debug cases."
            )
        ),
        "whatImproves": "Yield recovery, reduced debug cycle time, and improved coverage when action is applied.",
        "shouldApprove": "Yes with regression gate" if kpi["severity"] in ("critical", "high") else "Review with stakeholders",
    }

    raw_rows = []
    for c in cases[:12]:
        conf = _case_confidence(c)
        rec_row = build_recommendation_row(c, 1)
        isolation = c.get("break_isolation_result") or ""
        raw_rows.append(
            {
                "pattern": f"PAT_{c.get('die_label', 'SCAN')}",
                "chain": c.get("chain_display") or c.get("chain_name") or c.get("timing_chain") or c.get("lot_folder", c["lot_id"]),
                "vector": c.get("die_label", ""),
                "cell": (
                    isolation
                    or c.get("candidate_cell")
                    or c.get("timing_flop")
                    or f"({int(c.get('wafer_x', 0))},{int(c.get('wafer_y', 0))})"
                ),
                "clock": "CLK_CAP" if c["min_slack"] < 9999 else "CLK_CORE",
                "coverage": round(96.0 - min(5.0, c["mismatch_count"] * 0.05), 1),
                "fault": c["fault_models"][0] if c.get("fault_models") else c.get("defect_type", "SA0"),
                "runtimeMs": c.get("fail_count") or c["mismatch_count"],
                "powerMw": 180 + c["cell_count"] * 12,
                "confidence": conf,
                "recommendationScore": conf,
                "recommendedAction": rec_row["recommendation"],
                "breakIsolationResult": isolation or None,
            }
        )

    diagnosis_results = []
    how_to_implement = None
    if kpi_id == "broken_chains":
        how_to_implement = BROKEN_CHAINS_HOW_TO_IMPLEMENT
        diagnosis_results = build_broken_chain_diagnosis_results(cases)
        # Lean payload — How to Implement + Output only (no decision/what chrome)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "topology",
            "vizSeries": [],
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [],
            "howToImplement": how_to_implement,
            "diagnosisResults": [
                {
                    "result": r["result"],
                    "lotId": r.get("lotId"),
                    "dieLabel": r.get("dieLabel"),
                    "chain": r.get("chain"),
                    "chainName": r.get("chainName"),
                    "candidateBit": r.get("candidateBit"),
                    "cellLabel": r.get("cellLabel"),
                    "scanLength": r.get("scanLength") or 234,
                    "scanIn": r.get("scanIn"),
                    "scanOut": r.get("scanOut"),
                }
                for r in diagnosis_results
            ],
            "layout": "broken_chains_clean",
        }

    if kpi_id == "debug_recommendations":
        rec_rows = build_scan_chain_debug_recommendations(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "priority_matrix",
            "vizSeries": [],
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [],
            "howToImplement": None,
            "diagnosisResults": rec_rows,
            "layout": "scan_chain_recs_clean",
        }

    if kpi_id == "avg_ai_confidence":
        conf_rows = build_scan_chain_confidence_results(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "gauge",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [],
            "howToImplement": None,
            "diagnosisResults": conf_rows,
            "layout": "scan_chain_confidence_clean",
        }

    if kpi_id == "constraint_violations":
        violation_rows = build_constraint_violation_results(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "dependency",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [
                "How many reset vs scan-enable vs clock violations?",
                "Which fan-out cones are under reset constraint?",
                "How to review ATPG constraints?",
            ],
            "howToImplement": (
                "Classify STIL held pins into Reset (RESET_N), Scan Enable (TEST), and Clock (XI). "
                "Each typed held pin × failing fan-out cluster is one suspected over-constraint "
                "for ATPG review (not proven without raw ATPG DRC log)."
            ),
            "diagnosisResults": violation_rows,
            "layout": "constraint_violations_clean",
        }

    if kpi_id == "pending_review":
        review_rows = build_constraint_review_recommendations(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "assignment",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [
                "Show reset vs scan-enable vs clock review recommendations",
                "Which historical resolutions apply?",
                "What should we relax to don't-care?",
            ],
            "howToImplement": (
                "Same STIL held-pin × failing fan-out evidence as Constraint Violations, "
                "plus historical constraint-masked outcomes. Emits category-specific review "
                "text (Reset / Scan Enable / Clock) with recommended action and historical cite."
            ),
            "diagnosisResults": review_rows,
            "layout": "constraint_review_recs_clean",
        }

    if kpi_id == "coverage_impact":
        impact_rows = build_coverage_impact_results(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "heatmap",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [
                "Which constraint signature has the highest failing-pattern share?",
                "Show reset vs scan-enable vs clock coverage impact",
                "Is this estimate only?",
            ],
            "howToImplement": (
                "Whole-data Coverage Impact = unique failing patterns touched by any ATPG "
                "constraint signature / all unique failing patterns in fail-die logs. "
                "Also reports Reset / Scan Enable / Clock category shares and per-signature "
                "shares. Estimate only — not ATPG fault coverage."
            ),
            "diagnosisResults": impact_rows,
            "layout": "coverage_impact_clean",
        }

    if kpi_id == "timing_violations":
        timing_rows = build_timing_violation_results(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "timing",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [
                "Which patterns fail only at the fast timing set?",
                "Show setup vs hold at-speed fails",
                "What is STIL capture edge spacing?",
            ],
            "howToImplement": (
                "Compare fail logs across timing sets for the same pattern; "
                "if fail only appears above a frequency threshold, flag as at-speed/"
                "timing-correlated. Use STIL WaveformTable edge deltas as relative "
                "margin proxy. When only one insertion exists, slower set = Period×2 "
                "half-rate reference from STIL."
            ),
            "diagnosisResults": timing_rows,
            "layout": "timing_violations_clean",
        }

    if kpi_id == "timing_debug_recs":
        rec_rows = build_timing_debug_recommendations(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "clock_tree",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [
                "Show capture-window review recommendations",
                "Which patterns cite the most historical frequency fails?",
                "Which diagnosis transition/path-delay flags apply?",
            ],
            "howToImplement": (
                "Same at-speed timing evidence as Timing Violations (fail logs + STIL "
                "WaveformTable), plus diagnosis transition/path-delay flags and historical "
                "cases. Template names pattern / chain / clock domain and cites similar "
                "historical timing fails with frequency."
            ),
            "diagnosisResults": rec_rows,
            "layout": "timing_debug_recs_clean",
        }

    if kpi_id == "worst_slack":
        slack_rows = build_worst_slack_results(cases)
        summary = get_worst_slack_summary(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "critical_path",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [
                "What is the frequency margin proxy?",
                "Show worst slack in ps",
                "Which pattern has the worst margin?",
            ],
            "howToImplement": (
                "From at-speed fail vs pass timing-set frequencies: "
                "margin ≈ (f_fail − f_pass) / f_fail. Append diagnosis worst slack (ps) "
                "when available. Example: Fails at 800MHz, passes at 650MHz — ~19% "
                "frequency margin proxy; worst slack −47 ps."
            ),
            "diagnosisResults": slack_rows,
            "layout": "worst_slack_clean",
            "worstSlackSummary": summary,
        }

    if kpi_id == "power_violations":
        power_rows = build_power_violation_workspace_results(cases)
        power_summary = get_power_violation_summary(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "power",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [
                "Which patterns exceed IR drop threshold?",
                "Show PASS patterns flagged for thermal",
                "What are the IR and thermal thresholds?",
            ],
            "howToImplement": (
                "Scan IR_DROP_MV and THERMAL_C for every pattern in the ATE run. "
                f"Flag when IR_DROP_MV > {power_summary.get('irThresholdMv', 15)}mV or "
                f"THERMAL_C > {power_summary.get('thermalThresholdC', 55)}°C, even if STATUS=PASS "
                "(marginal patterns often pass but are close to failing). "
                "Drill-down lists top findings by severity; card count is full."
            ),
            "diagnosisResults": power_rows,
            "layout": "power_violations_clean",
            "powerViolationSummary": power_summary,
        }

    if kpi_id == "power_debug_recs":
        rec_rows = build_power_debug_recs_workspace(cases)
        rec_count = count_power_debug_recommendations(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "power",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [
                "Show IR-drop capture check recommendations",
                "Which patterns cite the most historical IR-drop fails?",
                "Which recommendations are flagged despite PASS?",
            ],
            "howToImplement": (
                "Same IR_DROP_MV / THERMAL_C evidence as Power Violations, plus past "
                "IR-drop signatures that led to STATUS=FAIL elsewhere and CENTER/NEAR_FULL "
                "historical cases. Template names pattern, measured value, % above threshold, "
                "and cites similar IR levels that preceded test-only fails within ±5 patterns "
                "— recommend monitoring adjacent patterns. "
                f"Card shows {rec_count} recommendations; list is top severity sample."
            ),
            "diagnosisResults": rec_rows,
            "layout": "power_debug_recs_clean",
            "powerDebugRecSummary": {
                "count": rec_count,
                "workspaceRows": len(rec_rows),
            },
        }

    if kpi_id == "peak_switching":
        peak_rows = build_peak_switching_workspace(cases)
        peak_summary = get_peak_switching_summary(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "heatmap",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [
                "What is the peak IR-drop pattern?",
                "How does peak compare to run average?",
                "Show highest switching-proxy patterns",
            ],
            "howToImplement": (
                "Use IR_DROP_MV as the direct switching-activity proxy (electrical "
                "downstream of toggle activity — no STIL toggle count needed). "
                "Take MAX(IR_DROP_MV) across all patterns with pattern reference, "
                "and compare to the run average. Example: Peak IR-drop: 20mV at "
                "Pattern #4521 (vs. avg 8mV across run)."
            ),
            "diagnosisResults": peak_rows,
            "layout": "peak_switching_clean",
            "peakSwitchingSummary": peak_summary,
        }

    if kpi_id == "defect_suspects":
        suspect_rows = build_defect_suspect_workspace(cases)
        suspect_summary = get_defect_suspect_summary(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "wafer",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [
                "Show top ranked defect suspects",
                "Which nets have the best pattern consistency?",
                "How is consistency computed from the failing bitmap?",
            ],
            "howToImplement": (
                "Pull top-N ranked candidates from diagnosis (FR-002). Validate each by "
                "checking what fraction of failing patterns are consistent with that "
                "candidate being the true fault site (corroborating/observations). "
                "Resolve cell/net labels via diagnosis Exact Cell Name + STIL scan-chain "
                "order for (Ua→Ub). Example: Net N4521 (U890→U912) — diagnosis rank 1, "
                "consistent with 11/12 failing patterns; Net N4487 — rank 2, consistent "
                "with 6/12."
            ),
            "diagnosisResults": suspect_rows,
            "layout": "defect_suspects_clean",
            "defectSuspectSummary": suspect_summary,
        }

    if kpi_id == "investigation_recs":
        rec_rows = build_investigation_recs_workspace(cases)
        rec_summary = get_investigation_recs_summary(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "history",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [
                "Show investigation recommendations with PFA technique",
                "Which nets ruled out power-induced false fails?",
                "What transition-fault / IR-drop cross-check was used?",
            ],
            "howToImplement": (
                "Start from Defect Suspects (top ranked nets). Cross-check "
                "TRANSITION_FAULTS vs IR_DROP_MV to rule out power-induced false fails "
                f"(normal IR below {rec_summary.get('irThresholdMv', 15)}mV with TF present). "
                "Cite fault-type hypothesis and recommend PFA technique from historical "
                "cases with matching diagnosis signature. Example: Investigate Net N4521 "
                "(U890→U912) — suspected bridging fault. Confirmed real defect: transition "
                "faults present (count=3) with normal IR-drop (8mV, below 15mV threshold), "
                "ruling out power-induced false fail. 3 historical cases with matching "
                "diagnosis signature were resolved by SEM → FIB → PFA confirming bridging short."
            ),
            "diagnosisResults": rec_rows,
            "layout": "investigation_recs_clean",
            "investigationRecSummary": rec_summary,
        }

    if kpi_id == "defect_localization":
        loc_rows = build_defect_localization_workspace(cases)
        loc_summary = get_defect_localization_summary(cases)
        return {
            "kpiId": kpi_id,
            "title": kpi["title"],
            "decision": {
                "executiveSummary": "",
                "rootCause": "",
                "confidence": agent_confidence,
                "businessImpact": "",
                "risk": "",
                "recommendation": "",
                "whatFailed": "",
                "whyAiRecommended": "",
                "whatImproves": "",
                "shouldApprove": "",
            },
            "summaryCards": [],
            "visualizationType": "wafer",
            "vizSeries": _build_viz_series(cases, kpi_id),
            "breakdown": [],
            "impact": [],
            "timeline": [],
            "rawRows": [],
            "copilotStarters": [
                "What is the average localization confidence?",
                "Which nets have the highest localization confidence?",
                "Show FR-009 die-local / wafer coordinates",
            ],
            "howToImplement": (
                "Combine Defect Suspects + Investigation Recommendations with FR-009 "
                "debug locations (die-local µm / wafer XY / priority). Score = diagnosis "
                "confidence × pattern consistency × TF/IR power cross-check × historical "
                "PFA boost × XY availability × debug priority. KPI card shows average "
                f"confidence ({loc_summary.get('kpiValue', 'N/A')}). Example: Defect "
                "localization confidence: 87% — Net N166 at wafer (130.2, 84.1), rank 1, "
                "consistency 4/4, PFA precedent 5, priority High."
            ),
            "diagnosisResults": loc_rows,
            "layout": "defect_localization_clean",
            "defectLocalizationSummary": loc_summary,
        }

    copilot_by_kpi = {
        "broken_chains": [
            "Which chains are broken?",
            "Show break isolation results",
            "How to implement chain diagnosis?",
        ],
        "timing_violations": [
            "Which patterns fail only at the fast timing set?",
            "Show setup vs hold at-speed fails",
            "What is STIL capture edge spacing?",
        ],
        "timing_debug_recs": [
            "Show capture-window review recommendations",
            "Which patterns cite the most historical frequency fails?",
            "Which diagnosis transition/path-delay flags apply?",
        ],
        "power_debug_recs": [
            "Show IR-drop capture check recommendations",
            "Which patterns cite the most historical IR-drop fails?",
            "Which recommendations are flagged despite PASS?",
        ],
        "defect_suspects": [
            "Show top ranked defect suspects",
            "Which nets have the best pattern consistency?",
            "How is consistency computed from the failing bitmap?",
        ],
        "investigation_recs": [
            "Show investigation recommendations with PFA technique",
            "Which nets ruled out power-induced false fails?",
            "What transition-fault / IR-drop cross-check was used?",
        ],
        "defect_localization": [
            "What is the average localization confidence?",
            "Which nets have the highest localization confidence?",
            "Show FR-009 die-local / wafer coordinates",
        ],
    }
    default_starters = [
        "Why was this recommendation generated?",
        "Explain root cause.",
        "Compare similar failures.",
        "Estimate yield improvement.",
    ]

    return {
        "kpiId": kpi_id,
        "title": kpi["title"],
        "decision": decision,
        "summaryCards": _build_summary_cards(kpi, cases, dataset),
        "visualizationType": KPI_VIZ_TYPES.get(kpi_id, "topology"),
        "vizSeries": _build_viz_series(cases, kpi_id),
        "breakdown": _build_breakdown(cases, kpi_id),
        "impact": _build_impact(kpi_id, cases),
        "timeline": [
            {"id": "gen", "label": "Generated", "at": "Today", "status": "done"},
            {"id": "rev", "label": "Reviewed", "at": "—", "status": "active"},
            {"id": "apr", "label": "Approved", "at": "—", "status": "upcoming"},
        ],
        "rawRows": raw_rows,
        "copilotStarters": copilot_by_kpi.get(kpi_id, default_starters),
        "howToImplement": how_to_implement,
        "diagnosisResults": diagnosis_results,
        "layout": "default",
    }
