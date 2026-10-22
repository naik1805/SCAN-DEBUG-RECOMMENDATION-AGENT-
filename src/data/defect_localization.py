"""
Defect Localization KPI (Physical Defect Investigation).

Inputs:
  Defect Suspects + Investigation Recommendations (analyzed/predicted paths)
  + FR-009 Debug Location Recommendations (die-local µm / wafer XY / priority)
  + failing-bitmap wafer XY fallback

Output: localization confidence % from weighted diagnosis × consistency ×
power false-fail cross-check × historical PFA × XY availability × priority.

Example:
  Defect localization confidence: 87% — Net N166 (U164→U166) at wafer (130.2, 84.1),
  rank 1, consistency 4/4, PFA precedent 5, priority High
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from src.data.dataset_builder import (
    _TableCollector,
    _find_table_by_header,
    _header_index,
    _row_get,
)
from src.data.defect_suspects import clear_defect_suspects_caches
from src.data.investigation_recs import (
    build_investigation_recommendations,
    clear_investigation_recs_caches,
)
from src.data.paths import (
    COMPILED_DATASET_PATH,
    DEFECT_LOCALIZATION_PATH,
    FAILING_BITMAPS_PATH,
    diagnosis_html_path,
)

MAX_CONFIDENCE_SCORE = 0.99
_TOP_N = 25
_MAX_WORKSPACE_ROWS = 40

_ROWS_CACHE: Optional[Tuple[str, List[Dict[str, Any]]]] = None
_SUMMARY_CACHE: Optional[Tuple[str, Dict[str, Any]]] = None

_WAFER_XY_RE = re.compile(
    r"([A-Za-z0-9_]+)\s*/\s*([A-Za-z0-9_]+)\s*:\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
)


def _inputs_fingerprint() -> str:
    parts: List[str] = ["v1-defect-localization-conf"]
    diag = diagnosis_html_path()
    for path in (diag, FAILING_BITMAPS_PATH, COMPILED_DATASET_PATH):
        if path and os.path.exists(path):
            parts.append(f"{path}:{os.path.getmtime(path)}")
    return "|".join(parts)


def _parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").replace("µm", "").replace("um", "").strip())
    except (TypeError, ValueError):
        digits = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(digits.group(0)) if digits else None


def _parse_wafer_occurrence(blob: str) -> Optional[Dict[str, Any]]:
    """Take first LOT/die:(x,y) from FR-009 Wafer Occurrences cell."""
    if not blob:
        return None
    m = _WAFER_XY_RE.search(blob.replace("\n", " "))
    if not m:
        return None
    return {
        "lotId": m.group(1),
        "dieLabel": m.group(2),
        "waferX": float(m.group(3)),
        "waferY": float(m.group(4)),
    }


@lru_cache(maxsize=1)
def _load_fr009_index() -> Tuple[Dict[str, Any], ...]:
    """FR-009 debug location recommendations (die-local + wafer XY + priority)."""
    path = diagnosis_html_path()
    if not path or not os.path.exists(path):
        return tuple()
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except OSError:
        return tuple()

    try:
        parser = _TableCollector()
        parser.feed(html)
        table = _find_table_by_header(parser.tables, "die local") or _find_table_by_header(
            parser.tables, "debug priority"
        )
        if not table:
            return tuple()
        idx = _header_index(table[0])
        rows: List[Dict[str, Any]] = []
        for row in table[1:]:
            cell = (_row_get(row, idx, "scan", "cell") or "").strip() or None
            chain = (_row_get(row, idx, "scan", "chain") or "").strip() or None
            ff = (_row_get(row, idx, "fail", "flop") or "").strip() or None
            if not cell and not ff:
                continue
            die_x = _parse_float(_row_get(row, idx, "die", "local", "x"))
            die_y = _parse_float(_row_get(row, idx, "die", "local", "y"))
            wafer_blob = _row_get(row, idx, "wafer", "occurrences") or ""
            wafer = _parse_wafer_occurrence(wafer_blob)
            pri_raw = (_row_get(row, idx, "debug", "priority") or "").strip()
            priority = "Medium"
            low = pri_raw.lower()
            if "high" in low:
                priority = "High"
            elif "low" in low:
                priority = "Low"
            rows.append(
                {
                    "cellName": cell,
                    "chain": chain,
                    "flipFlopId": ff,
                    "dieLocalXUm": die_x,
                    "dieLocalYUm": die_y,
                    "waferX": wafer.get("waferX") if wafer else None,
                    "waferY": wafer.get("waferY") if wafer else None,
                    "waferLotId": wafer.get("lotId") if wafer else None,
                    "waferDieLabel": wafer.get("dieLabel") if wafer else None,
                    "debugPriority": priority,
                }
            )
        return tuple(rows)
    except Exception:
        return tuple()


def _fr009_match(inv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Join investigation/suspect row to FR-009 by cell, FF id, or chain+bit."""
    cell = str(inv.get("cellName") or "").strip()
    ff = str(inv.get("flipFlopId") or "").strip()
    chain = str(inv.get("chain") or "").strip().lower()
    # net N166 ↔ FF_166
    net = str(inv.get("netId") or "")
    net_digits = "".join(ch for ch in net if ch.isdigit())

    best: Optional[Dict[str, Any]] = None
    for row in _load_fr009_index():
        score = 0
        if cell and row.get("cellName") == cell:
            score += 3
        if ff and row.get("flipFlopId") == ff:
            score += 3
        if net_digits and row.get("flipFlopId") and net_digits in str(row.get("flipFlopId")):
            score += 2
        if chain and str(row.get("chain") or "").lower() == chain:
            score += 1
        if score >= 3:
            return row
        if score > 0 and (best is None or score > int(best.get("_score") or 0)):
            tagged = dict(row)
            tagged["_score"] = score
            best = tagged
    if best and int(best.get("_score") or 0) >= 2:
        best.pop("_score", None)
        return best
    return None


def _priority_term(priority: Optional[str]) -> float:
    p = str(priority or "").lower()
    if p == "high":
        return 1.0
    if p == "low":
        return 0.85
    return 0.92


def _xy_term(fr009: Optional[Dict[str, Any]], inv: Dict[str, Any]) -> Tuple[float, bool]:
    if fr009 and fr009.get("dieLocalXUm") is not None and fr009.get("dieLocalYUm") is not None:
        return 1.0, True
    if fr009 and fr009.get("waferX") is not None and fr009.get("waferY") is not None:
        return 0.92, True
    if inv.get("waferX") is not None and inv.get("waferY") is not None:
        return 0.85, True
    return 0.55, False


def compute_localization_confidence(
    *,
    diagnosis_conf_pct: float,
    consistency_ratio: float,
    power_induced_ruled_out: bool,
    historical_match_count: int,
    xy_available: bool,
    debug_priority: Optional[str],
) -> Tuple[float, int]:
    """
    Weighted product → (score 0–0.99, pct).
    diagnosis × consistency × power × hist_boost × xy × priority
    """
    diagnosis_term = max(0.0, min(1.0, float(diagnosis_conf_pct) / 100.0))
    pattern_term = max(0.05, min(1.0, float(consistency_ratio)))
    power_term = 1.0 if power_induced_ruled_out else 0.85
    hist_boost = min(1.25, 1.0 + 0.05 * max(0, int(historical_match_count)))
    xy_term = 1.0 if xy_available else 0.55
    # If caller already folded priority into xy via _xy_term, still apply priority
    priority = _priority_term(debug_priority)

    score = min(
        MAX_CONFIDENCE_SCORE,
        max(0.0, diagnosis_term * pattern_term * power_term * hist_boost * xy_term * priority),
    )
    return round(score, 4), int(round(score * 100))


def format_defect_localization_result(
    *,
    confidence_pct: int,
    net_id: str,
    neighbor_from: Optional[str] = None,
    neighbor_to: Optional[str] = None,
    wafer_x: Optional[float] = None,
    wafer_y: Optional[float] = None,
    die_local_x: Optional[float] = None,
    die_local_y: Optional[float] = None,
    diagnosis_rank: Optional[int] = None,
    consistent: Optional[int] = None,
    total: Optional[int] = None,
    hist_count: Optional[int] = None,
    priority: Optional[str] = None,
) -> str:
    """
    Product-style:
      Defect localization confidence: 87% — Net N166 (U164→U166) at wafer (130.2, 84.1),
      rank 1, consistency 4/4, PFA precedent 5, priority High
    """
    if neighbor_from and neighbor_to:
        net_txt = f"Net {net_id} ({neighbor_from}→{neighbor_to})"
    else:
        net_txt = f"Net {net_id}"

    loc = ""
    if wafer_x is not None and wafer_y is not None:
        loc = f" at wafer ({wafer_x:.1f}, {wafer_y:.1f})"
    elif die_local_x is not None and die_local_y is not None:
        loc = f" at die-local ({die_local_x:.1f}µm, {die_local_y:.1f}µm)"

    extras: List[str] = []
    if diagnosis_rank is not None:
        extras.append(f"rank {diagnosis_rank}")
    if consistent is not None and total is not None:
        extras.append(f"consistency {consistent}/{total}")
    if hist_count is not None:
        extras.append(f"PFA precedent {hist_count}")
    if priority:
        extras.append(f"priority {priority}")

    head = f"Defect localization confidence: {confidence_pct}% — {net_txt}{loc}"
    if extras:
        return f"{head}, " + ", ".join(extras)
    return head


def _row_from_investigation(inv: Dict[str, Any]) -> Dict[str, Any]:
    fr009 = _fr009_match(inv)
    xy_term_val, xy_ok_from_coords = _xy_term(fr009, inv)
    # Prefer FR-009 priority; fold xy_term into compute via xy_available flag
    # but apply fr009 xy quality: if only wafer (0.92), scale after
    priority = (fr009 or {}).get("debugPriority") or "Medium"
    diagnosis_conf = float(inv.get("confidencePct") or 0)
    consistency = float(inv.get("consistencyRatio") or 0)
    if consistency <= 0 and inv.get("totalFailingPatterns"):
        c = int(inv.get("consistentPatterns") or 0)
        t = max(1, int(inv.get("totalFailingPatterns") or 1))
        consistency = c / t
    hist = int(inv.get("historicalMatchCount") or 0)
    ruled = bool(inv.get("powerInducedRuledOut"))

    score, pct = compute_localization_confidence(
        diagnosis_conf_pct=diagnosis_conf,
        consistency_ratio=consistency,
        power_induced_ruled_out=ruled,
        historical_match_count=hist,
        xy_available=xy_ok_from_coords,
        debug_priority=priority,
    )
    # Soften if only wafer XY (not die-local)
    if fr009 and fr009.get("dieLocalXUm") is None and fr009.get("waferX") is not None:
        score = min(MAX_CONFIDENCE_SCORE, round(score * 0.92 / 1.0, 4))
        pct = int(round(score * 100))
    elif not xy_ok_from_coords:
        pass
    else:
        # Re-apply xy_term_val nuance already via xy_available True = 1.0 for die-local
        if xy_term_val < 1.0:
            score = min(MAX_CONFIDENCE_SCORE, round(score * xy_term_val, 4))
            pct = int(round(score * 100))

    wafer_x = (fr009 or {}).get("waferX")
    wafer_y = (fr009 or {}).get("waferY")
    die_x = (fr009 or {}).get("dieLocalXUm")
    die_y = (fr009 or {}).get("dieLocalYUm")
    net_id = str(inv.get("netId") or "N?")
    consistent = inv.get("consistentPatterns")
    total = inv.get("totalFailingPatterns")
    result = format_defect_localization_result(
        confidence_pct=pct,
        net_id=net_id,
        neighbor_from=inv.get("neighborFrom"),
        neighbor_to=inv.get("neighborTo"),
        wafer_x=float(wafer_x) if wafer_x is not None else None,
        wafer_y=float(wafer_y) if wafer_y is not None else None,
        die_local_x=float(die_x) if die_x is not None else None,
        die_local_y=float(die_y) if die_y is not None else None,
        diagnosis_rank=inv.get("diagnosisRank") or inv.get("rank"),
        consistent=int(consistent) if consistent is not None else None,
        total=int(total) if total is not None else None,
        hist_count=hist,
        priority=priority,
    )

    return {
        "result": result,
        "rank": inv.get("rank"),
        "diagnosisRank": inv.get("diagnosisRank") or inv.get("rank"),
        "netId": net_id,
        "cellName": inv.get("cellName") or (fr009 or {}).get("cellName"),
        "flipFlopId": inv.get("flipFlopId") or (fr009 or {}).get("flipFlopId"),
        "neighborFrom": inv.get("neighborFrom"),
        "neighborTo": inv.get("neighborTo"),
        "chain": inv.get("chain") or (fr009 or {}).get("chain"),
        "confidencePct": pct,
        "confidenceScore": score,
        "diagnosisConfidencePct": diagnosis_conf,
        "consistencyRatio": round(consistency, 4),
        "consistentPatterns": consistent,
        "totalFailingPatterns": total,
        "powerInducedRuledOut": ruled,
        "historicalMatchCount": hist,
        "faultHypothesis": inv.get("faultHypothesis"),
        "dieLocalXUm": die_x,
        "dieLocalYUm": die_y,
        "waferX": wafer_x,
        "waferY": wafer_y,
        "waferLotId": (fr009 or {}).get("waferLotId"),
        "waferDieLabel": (fr009 or {}).get("waferDieLabel"),
        "debugPriority": priority,
        "xyAvailable": xy_ok_from_coords,
        "transitionFaultCount": inv.get("transitionFaultCount"),
        "irDropMv": inv.get("irDropMv"),
        "pfaTechnique": inv.get("pfaTechnique"),
        "recommendedAction": "INVESTIGATE_PHYSICAL_DEFECT",
        "recommendDetail": (
            "Localization confidence from diagnosis × pattern consistency × "
            "TF/IR cross-check × historical PFA × FR-009 XY/priority"
        ),
    }


def _build_summary_and_rows(
    cases: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    inv_rows = build_investigation_recommendations(cases, limit=max(_TOP_N, _MAX_WORKSPACE_ROWS))
    rows = [_row_from_investigation(r) for r in inv_rows]
    rows.sort(
        key=lambda r: (
            -float(r.get("confidenceScore") or 0),
            -int(r.get("historicalMatchCount") or 0),
            int(r.get("diagnosisRank") or 999),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    top = rows[:_TOP_N]
    if top:
        avg_score = sum(float(r.get("confidenceScore") or 0) for r in top) / len(top)
        avg_pct = int(round(avg_score * 100))
        kpi = f"{avg_pct}%"
        hero = top[0]["result"]
    else:
        avg_pct = 0
        kpi = "0%"
        hero = "N/A"

    by_pri = Counter(str(r.get("debugPriority") or "Medium") for r in top)
    summary = {
        "count": len(top),
        "kpiValue": kpi,
        "averageConfidencePct": avg_pct,
        "result": hero,
        "topNetId": top[0].get("netId") if top else None,
        "topConfidencePct": top[0].get("confidencePct") if top else None,
        "xyAvailableCount": sum(1 for r in top if r.get("xyAvailable")),
        "byPriority": dict(by_pri),
        "topN": _TOP_N,
    }
    return summary, rows


def build_defect_localization_results(
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


def build_defect_localization_workspace(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return build_defect_localization_results(cases, limit=_MAX_WORKSPACE_ROWS)


def get_defect_localization_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    global _SUMMARY_CACHE
    fingerprint = _inputs_fingerprint()
    if _SUMMARY_CACHE is not None and _SUMMARY_CACHE[0] == fingerprint:
        return _SUMMARY_CACHE[1]
    build_defect_localization_results(cases)
    if _SUMMARY_CACHE is not None and _SUMMARY_CACHE[0] == fingerprint:
        return _SUMMARY_CACHE[1]
    return {}


def average_defect_localization_confidence(cases: List[Dict[str, Any]]) -> str:
    summary = get_defect_localization_summary(cases)
    return str(summary.get("kpiValue") or "0%")


def warm_defect_localization_cache(cases: Optional[List[Dict[str, Any]]] = None) -> int:
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return 0
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if isinstance(cases, dict):
            cases = cases.get("cases") or cases.get("dataset") or []
    summary = get_defect_localization_summary(cases or [])
    return int(summary.get("count") or 0)


def clear_defect_localization_caches() -> None:
    global _ROWS_CACHE, _SUMMARY_CACHE
    _ROWS_CACHE = None
    _SUMMARY_CACHE = None
    _load_fr009_index.cache_clear()
    clear_investigation_recs_caches()
    clear_defect_suspects_caches()


def _load_persisted(
    fingerprint: str,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    if not os.path.exists(DEFECT_LOCALIZATION_PATH):
        return None
    try:
        with open(DEFECT_LOCALIZATION_PATH, encoding="utf-8") as f:
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
        os.makedirs(os.path.dirname(DEFECT_LOCALIZATION_PATH), exist_ok=True)
        with open(DEFECT_LOCALIZATION_PATH, "w", encoding="utf-8") as f:
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
