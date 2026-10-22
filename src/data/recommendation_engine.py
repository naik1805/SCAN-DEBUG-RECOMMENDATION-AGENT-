"""
Evidence-based scan debug recommendations for five responsibility areas:
  - Scan chain debug
  - ATPG constraint review
  - Timing debug
  - Power-related debug
  - Physical defect investigation
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.data.action_labels import CATEGORY_COLORS, action_label_from_internal, category_from_internal
from src.data.paths import HISTORICAL_CASES_PATH


def _clean_key(value: str) -> str:
    return str(value).strip().upper().replace("_", "").replace(" ", "").replace(".LOG", "")

SCAN_CHAIN = "INSPECT_SCAN_CHAIN"
ATPG_CONSTRAINT = "REVIEW_ATPG_CONSTRAINTS"
TIMING_DEBUG = "REVIEW_CAPTURE_CLOCK_TIMING"
POWER_DEBUG = "CHECK_IR_DROP_DURING_CAPTURE"
PHYSICAL_DEFECT = "INVESTIGATE_PHYSICAL_DEFECT"

CATEGORY_DISPLAY = {
    "SCAN_CHAIN_DEBUG": "Broken Chain",
    "TIMING_DEBUG": "Timing",
    "POWER_RELATED_DEBUG": "Power",
    "ATPG_CONSTRAINT_REVIEW": "ATPG Constraint",
    "PHYSICAL_DEFECT_INVESTIGATION": "Physical Defect",
}

PRIORITY_DISPLAY = {"P0": "Critical", "P1": "High", "P2": "Medium", "P3": "Low"}


def _pct_to_float(value: str) -> float:
    try:
        return float(str(value).replace("%", "").strip()) / 100.0
    except ValueError:
        return 0.0


def _case_confidence(case: Dict[str, Any]) -> float:
    base = 0.62
    if case.get("break_confidence_pct") is not None:
        base += min(0.25, float(case["break_confidence_pct"]))
    hist = case.get("historical_match") or {}
    if hist.get("confidence"):
        base += min(0.12, float(hist["confidence"]) * 0.12)
    if case.get("has_break"):
        base += 0.06
    if case.get("min_slack", 9999) < 9999:
        slack = case["min_slack"]
        base += 0.08 if slack < 0 else 0.04
    if case.get("mismatch_count", 0) >= 50:
        base += 0.04
    return round(min(0.99, base), 2)


def _priority_for_case(case: Dict[str, Any]) -> Tuple[str, str]:
    action = case.get("true_action", "")
    if action == SCAN_CHAIN and (case.get("shifter_failure") or case.get("fail_count", 0) >= 50):
        return "P0", PRIORITY_DISPLAY["P0"]
    if action == TIMING_DEBUG and case.get("min_slack", 9999) < 0:
        return "P0", PRIORITY_DISPLAY["P0"]
    if action == SCAN_CHAIN and case.get("has_break"):
        return "P1", PRIORITY_DISPLAY["P1"]
    if action in (TIMING_DEBUG, POWER_DEBUG) or case.get("min_slack", 9999) < 9999:
        return "P1", PRIORITY_DISPLAY["P1"]
    if action == PHYSICAL_DEFECT and case.get("defect_type") in ("LOCAL", "EDGE_LOC", "EDGE_RING"):
        return "P1", PRIORITY_DISPLAY["P1"]
    return "P2", PRIORITY_DISPLAY["P2"]


def _scan_chain_recommendation(case: Dict[str, Any]) -> Tuple[str, str, str]:
    chain = case.get("chain_name") or "suspect chain"
    bit = case.get("candidate_bit")
    cell = case.get("candidate_cell")
    pins = case.get("scan_pins")
    status = case.get("location_status", "UNKNOWN")
    failing_channels = case.get("failing_channels") or []
    chain_in_log = chain.lower() in [c.lower() for c in failing_channels]

    if bit and cell:
        rec = f"Inspect {chain} - isolate break at bit {bit} ({cell})"
    elif chain:
        rec = f"Inspect {chain} - trace shift/capture continuity"
    else:
        rec = "Inspect Scan Chain - trace shift/capture continuity"

    if case.get("shifter_failure"):
        root = f"Shift-phase failure on {chain} ({status}) - {case.get('fail_count', case.get('mismatch_count', 0))} failing patterns"
    else:
        root = f"Chain break suspected on {chain} - {status}, {case.get('fail_count', case.get('mismatch_count', 0))} pattern mismatches"

    if chain_in_log:
        root += f"; log confirms failures on {chain}"
    elif failing_channels:
        root += f"; log failures on {', '.join(failing_channels[:3])}"
    if pins:
        root += f"; scan pins {pins}"

    impact = "Restore chain integrity and recover failing patterns"
    if case.get("fail_count", 0) >= 50:
        impact = f"Clear {case['fail_count']} chain failures after re-stitch/isolation"

    return rec, root, impact


def _atpg_recommendation(case: Dict[str, Any]) -> Tuple[str, str, str]:
    lot = case.get("lot_id", "lot")
    die = case.get("die_label", "die")
    mismatches = case.get("mismatch_count", 0)
    test_mode = case.get("test_mode", "SCAN")
    patterns = case.get("total_patterns", 1000)

    rec = f"Review ATPG constraints for {lot}/{die} - {mismatches} mismatches without localized chain break"
    root = (
        f"Unclassified scan mismatch on {case.get('defect_type', 'NORMAL')} die - "
        f"constraint/mask conflict suspected in {test_mode} ({patterns} patterns)"
    )
    if case.get("has_break"):
        root += "; chain break present but defect signature points to ATPG coverage gap"
    impact = f"+{round(0.6 + mismatches * 0.02, 1)}% ATPG coverage after constraint fix"
    return rec, root, impact


def _timing_recommendation(case: Dict[str, Any]) -> Tuple[str, str, str]:
    slack = case.get("min_slack", 9999)
    chain = case.get("timing_chain") or case.get("chain_name") or "capture path"
    flop = case.get("timing_flop") or case.get("candidate_cell") or "capture flop"
    faults = case.get("fault_models") or []
    violations = case.get("timing_violations") or []

    slack_text = f"{slack:.0f} ps" if slack < 9999 else "marginal"
    fault_label = faults[0].replace("_", " ").title() if faults else "Capture Timing Violation"
    rec = f"Review capture clock timing on {chain} - fix {fault_label.lower()} at {flop} ({slack_text} slack)"

    detail = case.get("timing_detail")
    if detail:
        root = detail
    elif slack < 0:
        root = f"Hold/setup violation ({slack_text}) on {chain} - {', '.join(faults[:3]) or 'capture timing fault'}"
    else:
        root = f"Capture timing marginal on {chain} - {', '.join(faults[:3]) or 'review launch/capture skew'}"

    if len(violations) > 1:
        root += f"; {len(violations)} capture violations logged in diagnosis registry"

    impact = f"{abs(int(slack))} ps slack recovery" if slack < 9999 else "Improve capture margin"
    return rec, root, impact


def _power_recommendation(case: Dict[str, Any]) -> Tuple[str, str, str]:
    defect = case.get("defect_type", "CENTER")
    wx = case.get("wafer_x")
    wy = case.get("wafer_y")
    capture = case.get("capture_cycles", 2)
    test_mode = case.get("test_mode", "SCAN_ATSPEED")
    coord = f"({wx}, {wy})" if wx is not None and wy is not None else "wafer center cluster"

    rec = f"Check IR-drop during capture - {defect} defect at {coord}, {capture}-cycle capture in {test_mode}"
    root = (
        f"Center/near-full defect pattern ({defect}) indicates capture-phase IR-drop or "
        f"switching surge; {case.get('mismatch_count', 0)} mismatches during at-speed capture"
    )
    if case.get("chain_name"):
        root += f"; worst chain {case['chain_name']} may be secondary effect"
    impact = "Stabilize capture power domain and reduce false chain failures"
    return rec, root, impact


def _physical_recommendation(case: Dict[str, Any]) -> Tuple[str, str, str]:
    defect = case.get("defect_type", "LOCAL")
    wx = case.get("wafer_x")
    wy = case.get("wafer_y")
    cell = case.get("candidate_cell")
    hist = case.get("historical_match") or {}
    coord = f"({wx}, {wy})" if wx is not None and wy is not None else "localized cluster"

    suspect = hist.get("cell_name") or cell or "suspect cell from wafer map"
    chain = hist.get("chain") or case.get("chain_name")
    tool = "SEM → FIB → PFA" if hist.get("confidence", 0) >= 0.7 else "SEM localization → PFA"

    rec = f"Investigate physical defect - {defect} at {coord}, prioritize {suspect}"
    if chain:
        rec += f" on {chain}"

    root = f"{defect} defect signature at {coord}"
    if hist.get("predicted_root_cause"):
        root += f"; historical match confirms {hist['predicted_root_cause']} pattern"
    elif cell:
        root += f"; diagnosis candidate {cell}"
    else:
        root += "; localized failing bitmap - metal/via short suspected"

    if hist.get("confidence"):
        root += f" (historical confidence {hist['confidence'] * 100:.0f}%)"

    impact = f"+{round(0.8 + _case_confidence(case) * 1.5, 1)}% yield after PFA confirmation"
    return rec, root, impact


def _build_recommendation_text(case: Dict[str, Any]) -> Tuple[str, str, str]:
    action = case.get("true_action", ATPG_CONSTRAINT)
    builders = {
        SCAN_CHAIN: _scan_chain_recommendation,
        ATPG_CONSTRAINT: _atpg_recommendation,
        TIMING_DEBUG: _timing_recommendation,
        POWER_DEBUG: _power_recommendation,
        PHYSICAL_DEFECT: _physical_recommendation,
    }
    return builders.get(action, _atpg_recommendation)(case)


def _scan_chain_field(case: Dict[str, Any]) -> str:
    return case.get("chain_name") or case.get("timing_chain") or f"SC-{case.get('lot_id', 'X').replace('LOT_', '')}"


def build_recommendation_row(case: Dict[str, Any], index: int) -> Dict[str, Any]:
    category = category_from_internal(case.get("true_action", ""))
    rec_text, root_cause, impact = _build_recommendation_text(case)
    conf = _case_confidence(case)
    priority, priority_label = _priority_for_case(case)
    action_label = action_label_from_internal(case.get("true_action", ""))

    return {
        "id": f"DBG-REC-{index:03d}",
        "category": category,
        "categoryLabel": CATEGORY_DISPLAY.get(category, category),
        "actionLabel": action_label,
        "recommendation": rec_text,
        "scanChain": _scan_chain_field(case),
        "rootCause": root_cause,
        "priority": priority,
        "priorityLabel": priority_label,
        "affectedScanChain": case.get("lot_folder") or case.get("lot_id", ""),
        "expectedImpact": impact,
        "expectedYieldGainPct": round(0.4 + conf * 2, 1),
        "estimatedRuntimeReductionPct": round(1.0 + case.get("mismatch_count", 0) * 0.01, 1),
        "confidence": conf,
        "lotId": case.get("lot_id", ""),
        "dieLabel": case.get("die_label", ""),
        "responsibility": _responsibility_label(category),
        "evidence": _evidence_summary(case),
    }


def _responsibility_label(category: str) -> str:
    return {
        "SCAN_CHAIN_DEBUG": "Scan chain debug recommendations",
        "ATPG_CONSTRAINT_REVIEW": "ATPG constraint review recommendations",
        "TIMING_DEBUG": "Timing debug recommendations",
        "POWER_RELATED_DEBUG": "Power-related debug recommendations",
        "PHYSICAL_DEFECT_INVESTIGATION": "Physical defect investigation recommendations",
    }.get(category, category)


def _evidence_summary(case: Dict[str, Any]) -> str:
    parts = [
        f"{case.get('lot_id')}/{case.get('die_label')}",
        f"{case.get('mismatch_count', 0)} mismatches",
        case.get("defect_type", "NORMAL"),
    ]
    if case.get("chain_name"):
        parts.append(f"chain={case['chain_name']}")
    if case.get("failing_channel_count"):
        parts.append(f"{case['failing_channel_count']} failing channels")
    if case.get("candidate_cell"):
        parts.append(case["candidate_cell"])
    if case.get("min_slack", 9999) < 9999:
        parts.append(f"slack={case['min_slack']:.0f}ps")
    return " · ".join(str(p) for p in parts if p)


def _clamp_confidence(value: float) -> float:
    return round(min(0.99, max(0.50, float(value))), 2)


def _priority_tuple(code: str) -> Tuple[str, str]:
    return code, PRIORITY_DISPLAY.get(code, code)


def _row_shell(
    *,
    index: int,
    category: str,
    action_label: str,
    recommendation: str,
    scan_chain: str,
    root_cause: str,
    priority: str,
    expected_impact: str,
    confidence: float,
    lot_id: str = "",
    die_label: str = "",
    affected: str = "",
    evidence: str = "",
) -> Dict[str, Any]:
    conf = _clamp_confidence(confidence)
    prio, prio_label = _priority_tuple(priority)
    return {
        "id": f"DBG-REC-{index:03d}",
        "category": category,
        "categoryLabel": CATEGORY_DISPLAY.get(category, category),
        "actionLabel": action_label,
        "recommendation": recommendation,
        "scanChain": scan_chain or "—",
        "rootCause": root_cause,
        "priority": prio,
        "priorityLabel": prio_label,
        "affectedScanChain": affected or scan_chain or lot_id or "",
        "expectedImpact": expected_impact,
        "expectedYieldGainPct": round(0.4 + conf * 2, 1),
        "estimatedRuntimeReductionPct": round(1.0 + conf * 4, 1),
        "confidence": conf,
        "lotId": lot_id or "",
        "dieLabel": die_label or "",
        "responsibility": _responsibility_label(category),
        "evidence": evidence or recommendation,
    }


def _from_scan_chain_rec(row: Dict[str, Any]) -> Dict[str, Any]:
    chain_num = row.get("chainNumber")
    chain = (
        f"channel{chain_num}"
        if chain_num is not None
        else (row.get("chainName") or row.get("chain") or "scan chain")
    )
    bit = row.get("candidateBit")
    cell = row.get("cellLabel") or "suspect cell"
    fail = int(row.get("failCount") or 0)
    hist_n = int(row.get("historicalMatchCount") or 0)
    hist_sim = float(row.get("historicalSimilarity") or 0)
    rank = int(row.get("diagnosisRank") or 99)
    fault = row.get("faultType") or "stuck-at"

    recommendation = str(row.get("result") or "")
    if bit is not None:
        short = f"Inspect {chain} - isolate break at bit {bit} ({cell})"
    else:
        short = f"Inspect {chain} near cell {cell}"

    root = (
        f"Shift-phase / chain-break on {chain} - {fault} (diagnosis rank {rank}); "
        f"{fail} failing patterns"
    )
    if hist_n and hist_sim >= 0.78:
        root += (
            f"; {hist_n} historical signature match"
            f"{'es' if hist_n != 1 else ''} → {row.get('historicalRootCause') or 'scan path open'}"
        )
    pins = " / ".join(p for p in (row.get("scanIn"), row.get("scanOut")) if p)
    if pins:
        root += f"; scan pins {pins}"

    conf = 0.72 + min(0.14, hist_sim * 0.14) + (0.06 if fail >= 50 else 0.03)
    conf -= min(0.10, max(0, rank - 1) * 0.015)
    priority = "P0" if fail >= 50 or rank <= 3 else "P1"
    impact = (
        f"Clear {fail} chain failures after re-stitch/isolation"
        if fail
        else "Restore chain integrity and recover failing patterns"
    )
    return {
        "category": "SCAN_CHAIN_DEBUG",
        "actionLabel": "Inspect Scan Chain",
        "recommendation": recommendation or short,
        "scanChain": chain,
        "rootCause": root,
        "priority": priority,
        "expectedImpact": impact,
        "confidence": conf,
        "lotId": str(row.get("lotId") or ""),
        "dieLabel": str(row.get("dieLabel") or ""),
        "sortKey": (0 if priority == "P0" else 1, -conf, -fail),
    }


def _from_timing_rec(row: Dict[str, Any]) -> Dict[str, Any]:
    chain = str(row.get("timingChain") or "capture path")
    slack = row.get("worstSlackPs")
    try:
        slack_f = float(slack) if slack is not None else None
    except (TypeError, ValueError):
        slack_f = None
    hist_n = int(row.get("historicalMatchCount") or 0)
    kind = str(row.get("kind") or "timing")
    classification = str(row.get("classification") or "")
    flop = row.get("timingFlop") or "capture flop"
    domain = row.get("clockDomain") or "capture clock"

    recommendation = str(row.get("result") or "")
    detail = row.get("timingDetail") or row.get("recommendDetail") or ""
    if slack_f is not None and slack_f < 0:
        root = (
            f"{'Hold' if kind == 'hold' else 'Setup'} timing violation "
            f"(slack: {slack_f:.1f} ps) on {chain}; {classification or kind}"
        )
    else:
        root = detail or f"Capture timing review on {chain} / {domain} ({classification or kind})"
    if row.get("diagnosisTransitionPathDelay"):
        root += "; diagnosis flagged transition/path-delay"

    conf = 0.66 + min(0.12, hist_n * 0.025)
    if slack_f is not None and slack_f < 0:
        conf += 0.10
    elif slack_f is not None:
        conf += 0.04
    if row.get("diagnosisTransitionPathDelay"):
        conf += 0.04
    priority = "P0" if slack_f is not None and slack_f < 0 else "P1"
    impact = (
        f"{abs(int(round(slack_f)))} ps slack recovery"
        if slack_f is not None
        else "Improve capture-window margin"
    )
    return {
        "category": "TIMING_DEBUG",
        "actionLabel": "Review Capture Clock Timing",
        "recommendation": recommendation
        or f"Review capture clock timing on {chain} - fix at {flop}",
        "scanChain": chain,
        "rootCause": root,
        "priority": priority,
        "expectedImpact": impact,
        "confidence": conf,
        "lotId": str(row.get("lotId") or ""),
        "dieLabel": str(row.get("dieLabel") or ""),
        "sortKey": (
            0 if priority == "P0" else 1,
            -conf,
            slack_f if slack_f is not None else 0,
        ),
    }


def _from_power_rec(row: Dict[str, Any]) -> Dict[str, Any]:
    pattern_id = str(row.get("patternId") or "").lstrip("P") or "?"
    ir = row.get("irDropMv")
    thermal = row.get("thermalC")
    pct = float(row.get("pctAboveThreshold") or 0)
    hist_n = int(row.get("historicalMatchCount") or 0)
    flagged = bool(row.get("flaggedDespitePass"))
    chain = f"Pattern #{pattern_id}"

    recommendation = str(row.get("result") or "")
    if ir is not None:
        root = (
            f"IR-drop during capture on Pattern #{pattern_id}: {ir}mV "
            f"({pct:.0f}% above threshold)"
        )
    elif thermal is not None:
        root = f"Thermal elevation during capture on Pattern #{pattern_id}: {thermal}°C"
    else:
        root = f"Power integrity risk on Pattern #{pattern_id}"
    if flagged:
        root += "; flagged despite PASS status"
    if hist_n:
        root += f"; {hist_n} historical IR-fail signature match{'es' if hist_n != 1 else ''}"

    conf = 0.64 + min(0.14, pct / 100.0 * 0.25) + min(0.10, hist_n * 0.02)
    if flagged:
        conf += 0.05
    priority = "P0" if pct >= 20 or flagged or (ir is not None and float(ir) >= 50) else "P1"
    impact = (
        f"Mitigate IR-drop on Pattern #{pattern_id}; monitor ±5 adjacent patterns"
        if ir is not None
        else f"Stabilize capture power on Pattern #{pattern_id}"
    )
    return {
        "category": "POWER_RELATED_DEBUG",
        "actionLabel": "Check IR-Drop During Capture",
        "recommendation": recommendation
        or f"Check IR-drop during capture for Pattern #{pattern_id}",
        "scanChain": chain,
        "rootCause": root,
        "priority": priority,
        "expectedImpact": impact,
        "confidence": conf,
        "lotId": str(row.get("lotId") or ""),
        "dieLabel": str(row.get("dieLabel") or ""),
        "sortKey": (0 if priority == "P0" else 1, -conf, -pct),
    }


def _from_atpg_rec(row: Dict[str, Any]) -> Dict[str, Any]:
    label = str(row.get("constraintCategoryLabel") or row.get("constraintCategory") or "ATPG")
    pin = str(row.get("heldPins") or row.get("constraintPin") or "constraint pin")
    patterns = int(row.get("failingPatternCount") or 0)
    chains = int(row.get("chainsAffected") or 0)
    hist_n = int(row.get("historicalMatchCount") or 0)
    action = str(row.get("recommendedAction") or "Review")
    chain = (
        f"{chains} chain{'s' if chains != 1 else ''}"
        if chains
        else (row.get("fanoutSignal") or "ATPG constraint")
    )

    recommendation = str(row.get("result") or "")
    root = (
        f"Suspected {label} over-constraint: {pin} "
        f"({patterns} failing pattern{'s' if patterns != 1 else ''}"
        f"{f', {chains} chains' if chains else ''})"
    )
    if row.get("historicalResolution"):
        root += f"; historical resolution: {row['historicalResolution']}"

    conf = 0.68 + min(0.12, hist_n * 0.03) + min(0.08, patterns * 0.002)
    priority = "P0" if patterns >= 40 or chains >= 8 else "P1"
    impact = (
        f"+{round(0.6 + min(4.0, patterns * 0.02), 1)}% ATPG coverage after "
        f"{action.lower()} constraint fix"
    )
    return {
        "category": "ATPG_CONSTRAINT_REVIEW",
        "actionLabel": "Review ATPG Constraints",
        "recommendation": recommendation or f"{action} {label} constraint on {pin}",
        "scanChain": str(chain),
        "rootCause": root,
        "priority": priority,
        "expectedImpact": impact,
        "confidence": conf,
        "lotId": str(row.get("lotId") or ""),
        "dieLabel": str(row.get("dieLabel") or ""),
        "sortKey": (0 if priority == "P0" else 1, -conf, -patterns),
    }


def _from_physical_rec(row: Dict[str, Any]) -> Dict[str, Any]:
    net = str(row.get("netId") or "N?")
    chain_num = row.get("chainNumber")
    chain = (
        f"channel{chain_num}"
        if chain_num is not None
        else (row.get("chain") or net)
    )
    root_cause = str(row.get("rootCause") or row.get("faultHypothesis") or "physical defect")
    conf_pct = row.get("confidencePct")
    try:
        conf = float(conf_pct) / 100.0 if conf_pct is not None else 0.72
    except (TypeError, ValueError):
        conf = 0.72
    hist_n = int(row.get("historicalMatchCount") or 0)
    rank = int(row.get("rank") or row.get("diagnosisRank") or 99)
    if hist_n:
        conf = min(0.95, conf + min(0.08, hist_n * 0.02))
    if row.get("powerInducedRuledOut"):
        conf = min(0.97, conf + 0.04)

    recommendation = str(row.get("result") or "")
    nbr = ""
    if row.get("neighborFrom") and row.get("neighborTo"):
        nbr = f" ({row['neighborFrom']}→{row['neighborTo']})"
    root = f"{root_cause} on Net {net}{nbr}"
    if row.get("transitionFaultCount"):
        root += f"; TRANSITION_FAULTS={row['transitionFaultCount']}"
    if row.get("irDropMv") is not None:
        root += f"; IR-drop {row['irDropMv']}mV"

    priority = "P0" if rank <= 5 else "P1"
    impact = str(row.get("pfaTechnique") or "SEM localization → PFA confirmation")
    return {
        "category": "PHYSICAL_DEFECT_INVESTIGATION",
        "actionLabel": "Investigate Physical Defect",
        "recommendation": recommendation or f"Investigate Net {net}{nbr}",
        "scanChain": str(chain),
        "rootCause": root,
        "priority": priority,
        "expectedImpact": impact,
        "confidence": conf,
        "lotId": str(row.get("lotId") or ""),
        "dieLabel": str(row.get("dieLabel") or ""),
        "sortKey": (0 if priority == "P0" else 1, -conf, rank),
    }


def build_recommendations(dataset: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    """Top Scan Debug Recommendations — ML (default), KPI engines, or hybrid."""
    from src.config import get_settings

    settings = get_settings()
    src = settings.recommendation_source
    if limit is None:
        limit = settings.top_recommendations_limit

    if src == "kpi":
        return _build_kpi_recommendations(dataset, limit=limit)

    from src.data.ml_recommendations import build_ml_recommendations, build_recommendations_hybrid

    if src == "hybrid":
        return build_recommendations_hybrid(dataset, limit=limit)
    # default: ml with KPI text fallback inside build_ml_recommendations
    return build_ml_recommendations(dataset, limit=limit)


def _build_kpi_recommendations(dataset: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    """Top recommendations from live KPI engines only."""
    from src.data.atpg_constraint_review_recs import build_constraint_review_recommendations
    from src.data.investigation_recs import build_investigation_recommendations
    from src.data.power_debug_recs import build_power_debug_recommendations
    from src.data.scan_chain_debug_recs import build_scan_chain_debug_recommendations
    from src.data.timing_debug_recs import build_timing_debug_recommendations

    fail = [
        c
        for c in dataset
        if str(c.get("source_file") or "").startswith("fail")
        or str(c.get("lot_folder") or "").lower().startswith("fail")
    ]
    if not fail:
        fail = dataset

    # Balanced sample per responsibility area, then global re-rank
    per_category = max(2, (limit + 4) // 5)
    mapped: List[Dict[str, Any]] = []
    for src_rows, mapper in (
        (build_scan_chain_debug_recommendations(dataset)[:per_category], _from_scan_chain_rec),
        (build_timing_debug_recommendations(dataset)[:per_category], _from_timing_rec),
        (build_power_debug_recommendations(dataset)[:per_category], _from_power_rec),
        (build_constraint_review_recommendations(fail)[:per_category], _from_atpg_rec),
        (build_investigation_recommendations(dataset)[:per_category], _from_physical_rec),
    ):
        for row in src_rows:
            mapped.append(mapper(row))

    if not mapped:
        # Fallback: die-level stubs when engines return empty
        priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        sorted_cases = sorted(
            dataset,
            key=lambda c: (
                priority_rank.get(_priority_for_case(c)[0], 9),
                -_case_confidence(c),
                -c.get("mismatch_count", 0),
            ),
        )
        return [build_recommendation_row(case, i + 1) for i, case in enumerate(sorted_cases[:limit])]

    mapped.sort(key=lambda r: r.get("sortKey") or (9, 0))
    selected = mapped[:limit]
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(selected, start=1):
        out.append(
            _row_shell(
                index=i,
                category=item["category"],
                action_label=item["actionLabel"],
                recommendation=item["recommendation"],
                scan_chain=item["scanChain"],
                root_cause=item["rootCause"],
                priority=item["priority"],
                expected_impact=item["expectedImpact"],
                confidence=item["confidence"],
                lot_id=item.get("lotId", ""),
                die_label=item.get("dieLabel", ""),
                evidence=item["recommendation"],
            )
        )
    return out


def build_root_cause_distribution(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import Counter

    counts = Counter(category_from_internal(c.get("true_action", "")) for c in dataset)
    return [
        {
            "name": action_label_from_internal(
                {
                    "SCAN_CHAIN_DEBUG": SCAN_CHAIN,
                    "TIMING_DEBUG": TIMING_DEBUG,
                    "POWER_RELATED_DEBUG": POWER_DEBUG,
                    "ATPG_CONSTRAINT_REVIEW": ATPG_CONSTRAINT,
                    "PHYSICAL_DEFECT_INVESTIGATION": PHYSICAL_DEFECT,
                }.get(cat, ATPG_CONSTRAINT)
            ),
            "value": count,
            "fill": CATEGORY_COLORS.get(cat, "#7C3AED"),
        }
        for cat, count in counts.most_common()
    ]


def load_historical_index() -> Dict[str, List[Dict[str, Any]]]:
    if not os.path.exists(HISTORICAL_CASES_PATH):
        return {}
    with open(HISTORICAL_CASES_PATH, "r", encoding="utf-8") as f:
        doc = json.load(f)
    by_lot_chain: Dict[str, List[Dict[str, Any]]] = {}
    for item in doc.get("cases", []):
        chain = str(item.get("chain", "")).lower()
        if not chain:
            continue
        review = item.get("review") or {}
        lot_id = review.get("lot_id")
        if not lot_id:
            dedupe = str(review.get("dedupe_key", ""))
            parts = dedupe.split("|")
            if len(parts) >= 5:
                lot_id = parts[4]
        lot_key = _clean_key(lot_id) if lot_id else "*"
        by_lot_chain.setdefault(f"{lot_key}|{chain}", []).append(item)
        if lot_key != "*":
            by_lot_chain.setdefault(f"*||{chain}", []).append(item)
    return by_lot_chain


def match_historical_case(
    case: Dict[str, Any], hist_index: Dict[str, List[Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    chain = str(case.get("chain_name", "")).lower()
    if not chain:
        return None

    lot_key = _clean_key(case.get("lot_id", ""))
    candidate_cell = case.get("candidate_cell")
    pools = hist_index.get(f"{lot_key}|{chain}", []) + hist_index.get(f"*||{chain}", [])

    seen: set[str] = set()
    candidates: List[Dict[str, Any]] = []
    for item in pools:
        cid = str(item.get("case_id", ""))
        if cid in seen:
            continue
        seen.add(cid)
        candidates.append(item)

    if not candidates:
        return None

    def score(item: Dict[str, Any]) -> float:
        review = item.get("review") or {}
        base = float(item.get("confidence") or review.get("location_confidence") or 0)
        if candidate_cell and item.get("cell_name") == candidate_cell:
            base += 0.5
        if review.get("lot_id") and _clean_key(review.get("lot_id")) == lot_key:
            base += 0.2
        return base

    best = max(candidates, key=score)
    review = best.get("review") or {}
    return {
        "chain": best.get("chain"),
        "cell_name": best.get("cell_name"),
        "confidence": float(best.get("confidence") or review.get("location_confidence") or 0),
        "predicted_root_cause": best.get("predicted_root_cause") or review.get("predicted_root_cause"),
        "lot_id": review.get("lot_id"),
        "kind": best.get("kind"),
        "case_id": best.get("case_id"),
    }
