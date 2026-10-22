"""
ML-driven scan debug recommendations.

Uses a supervised gradient-boosting model to rank dies and pick the debug action;
KPI engines supply evidence-rich recommendation text when available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.config import get_settings
from src.data.action_labels import category_from_internal, action_label_from_internal
from src.data.recommendation_engine import (
    SCAN_CHAIN,
    TIMING_DEBUG,
    POWER_DEBUG,
    ATPG_CONSTRAINT,
    PHYSICAL_DEFECT,
    CATEGORY_DISPLAY,
    PRIORITY_DISPLAY,
    _build_recommendation_text,
    _priority_for_case,
    _scan_chain_field,
    _responsibility_label,
    _evidence_summary,
    _row_shell,
    _from_scan_chain_rec,
    _from_timing_rec,
    _from_power_rec,
    _from_atpg_rec,
    _from_physical_rec,
    build_recommendations as build_kpi_recommendations,
)
from src.models.supervised_recommender import (
    load_action_model,
    predict_action,
    INTERNAL_ACTIONS,
)

_ACTION_INTERNAL = {
    "SCAN_CHAIN_DEBUG": SCAN_CHAIN,
    "TIMING_DEBUG": TIMING_DEBUG,
    "POWER_RELATED_DEBUG": POWER_DEBUG,
    "ATPG_CONSTRAINT_REVIEW": ATPG_CONSTRAINT,
    "PHYSICAL_DEFECT_INVESTIGATION": PHYSICAL_DEFECT,
}

_KPI_SOURCES: Dict[str, Tuple[Any, Any]] = {}


def _lazy_kpi_sources(dataset: List[Dict[str, Any]], fail: List[Dict[str, Any]]) -> None:
    if _KPI_SOURCES:
        return
    from src.data.scan_chain_debug_recs import build_scan_chain_debug_recommendations
    from src.data.timing_debug_recs import build_timing_debug_recommendations
    from src.data.power_debug_recs import build_power_debug_recommendations
    from src.data.atpg_constraint_review_recs import build_constraint_review_recommendations
    from src.data.investigation_recs import build_investigation_recommendations

    _KPI_SOURCES[SCAN_CHAIN] = (
        build_scan_chain_debug_recommendations(dataset),
        _from_scan_chain_rec,
    )
    _KPI_SOURCES[TIMING_DEBUG] = (
        build_timing_debug_recommendations(dataset),
        _from_timing_rec,
    )
    _KPI_SOURCES[POWER_DEBUG] = (
        build_power_debug_recommendations(dataset),
        _from_power_rec,
    )
    _KPI_SOURCES[ATPG_CONSTRAINT] = (
        build_constraint_review_recommendations(fail),
        _from_atpg_rec,
    )
    _KPI_SOURCES[PHYSICAL_DEFECT] = (
        build_investigation_recommendations(dataset),
        _from_physical_rec,
    )


def _find_kpi_mapped_row(
    dataset: List[Dict[str, Any]],
    fail: List[Dict[str, Any]],
    case: Dict[str, Any],
    internal_action: str,
) -> Optional[Dict[str, Any]]:
    _lazy_kpi_sources(dataset, fail)
    if internal_action not in _KPI_SOURCES:
        return None
    rows, mapper = _KPI_SOURCES[internal_action]
    lot = str(case.get("lot_id") or "")
    die = str(case.get("die_label") or "")
    for row in rows:
        if str(row.get("lotId") or "") == lot and str(row.get("dieLabel") or "") == die:
            mapped = mapper(row)
            return mapped
    for row in rows:
        if str(row.get("lotId") or "") == lot:
            mapped = mapper(row)
            return mapped
    if rows:
        return mapper(rows[0])
    return None


def _urgency_score(case: Dict[str, Any]) -> float:
    mismatch = float(case.get("mismatch_count") or case.get("fail_count") or 0)
    base = min(mismatch / 70.0, 1.0)
    if case.get("has_break"):
        base += 0.15
    slack = float(case.get("min_slack", 9999))
    if slack < 0:
        base += 0.12
    return min(1.5, base)


def _ml_row_from_case(
    case: Dict[str, Any],
    internal_action: str,
    ml_confidence: float,
    prob_map: Dict[str, float],
    kpi_item: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if kpi_item:
        conf = round(min(0.99, 0.55 * ml_confidence + 0.45 * float(kpi_item.get("confidence") or 0.7)), 2)
        return {
            **kpi_item,
            "confidence": conf,
            "mlConfidence": round(ml_confidence, 3),
            "mlAction": internal_action,
            "mlProbabilities": {k: round(v, 3) for k, v in prob_map.items()},
            "recommendationSource": "ml+kpi",
        }

    category = category_from_internal(internal_action)
    rec_text, root_cause, impact = _build_recommendation_text_for_action(case, internal_action)
    priority, priority_label = _priority_for_case({**case, "true_action": internal_action})
    conf = round(min(0.99, ml_confidence), 2)
    root_cause = (
        f"ML model ({conf * 100:.0f}%): {action_label_from_internal(internal_action)} — {root_cause}"
    )
    return {
        "category": category,
        "actionLabel": action_label_from_internal(internal_action),
        "recommendation": rec_text,
        "scanChain": _scan_chain_field(case),
        "rootCause": root_cause,
        "priority": priority,
        "expectedImpact": impact,
        "confidence": conf,
        "lotId": str(case.get("lot_id") or ""),
        "dieLabel": str(case.get("die_label") or ""),
        "mlConfidence": round(ml_confidence, 3),
        "mlAction": internal_action,
        "mlProbabilities": {k: round(v, 3) for k, v in prob_map.items()},
        "recommendationSource": "ml",
        "sortKey": (
            0 if priority == "P0" else 1 if priority == "P1" else 2,
            -conf,
            -_urgency_score(case),
        ),
    }


def _build_recommendation_text_for_action(
    case: Dict[str, Any], internal_action: str
) -> Tuple[str, str, str]:
    from src.data.recommendation_engine import (
        _scan_chain_recommendation,
        _atpg_recommendation,
        _timing_recommendation,
        _power_recommendation,
        _physical_recommendation,
    )

    builders = {
        SCAN_CHAIN: _scan_chain_recommendation,
        ATPG_CONSTRAINT: _atpg_recommendation,
        TIMING_DEBUG: _timing_recommendation,
        POWER_DEBUG: _power_recommendation,
        PHYSICAL_DEFECT: _physical_recommendation,
    }
    return builders.get(internal_action, _atpg_recommendation)(case)


def build_ml_recommendations(
    dataset: List[Dict[str, Any]],
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Rank dies with supervised ML; attach KPI evidence text when available."""
    bundle = load_action_model()
    if bundle is None:
        return build_kpi_recommendations(dataset, limit=limit)

    fail = [
        c
        for c in dataset
        if str(c.get("source_file") or "").startswith("fail")
        or str(c.get("lot_folder") or "").lower().startswith("fail")
    ]
    if not fail:
        fail = list(dataset)

    scored: List[Tuple[float, Dict[str, Any], str, float, Dict[str, float]]] = []
    for case in dataset:
        action, conf, probs = predict_action(case, bundle=bundle)
        score = conf * _urgency_score(case)
        scored.append((score, case, action, conf, probs))

    scored.sort(key=lambda t: -t[0])
    selected = scored[:limit]

    out: List[Dict[str, Any]] = []
    for i, (_score, case, action, conf, probs) in enumerate(selected, start=1):
        kpi_item = _find_kpi_mapped_row(dataset, fail, case, action)
        item = _ml_row_from_case(case, action, conf, probs, kpi_item)
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
                evidence=_evidence_summary(case) + f" · ML {int(conf * 100)}%",
            )
        )
        out[-1]["mlConfidence"] = item.get("mlConfidence")
        out[-1]["recommendationSource"] = item.get("recommendationSource")
        out[-1]["mlProbabilities"] = item.get("mlProbabilities")

    return out


def build_recommendations_hybrid(
    dataset: List[Dict[str, Any]],
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """ML-ranked dies with KPI text; fill remaining slots from pure KPI top rows."""
    ml_rows = build_ml_recommendations(dataset, limit=limit)
    if len(ml_rows) >= limit:
        return ml_rows
    kpi_rows = build_kpi_recommendations(dataset, limit=limit)
    seen = {f"{r.get('lotId')}|{r.get('dieLabel')}|{r.get('recommendation')}" for r in ml_rows}
    for r in kpi_rows:
        key = f"{r.get('lotId')}|{r.get('dieLabel')}|{r.get('recommendation')}"
        if key not in seen and len(ml_rows) < limit:
            r = dict(r)
            r["recommendationSource"] = r.get("recommendationSource") or "kpi"
            ml_rows.append(r)
            seen.add(key)
    for j, row in enumerate(ml_rows, start=1):
        row["id"] = f"DBG-REC-{j:03d}"
    return ml_rows
