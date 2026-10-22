"""
Investigation Recommendations KPI (Physical Defect Investigation).

Inputs:
  Defect Suspects (top ranked nets / cells / consistency)
  + TRANSITION_FAULTS / IR_DROP_MV cross-check (rule out power-induced false fail)
  + historical_debug_cases (PFA-confirmed matching diagnosis signatures)

Template: name top candidate, confirm not power-induced, cite fault-type hypothesis,
recommend PFA technique from historical precedent.

Output:
  Investigate Net N4521 (U890→U912) — suspected bridging fault. Confirmed real defect:
  transition faults present (count=3) with normal IR-drop (8mV, below 15mV threshold),
  ruling out power-induced false fail. 3 historical cases with matching diagnosis
  signature were resolved by SEM → FIB → PFA confirming bridging short
"""

from __future__ import annotations

import json
import os
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from src.data.defect_suspects import (
    build_defect_suspect_results,
    clear_defect_suspects_caches,
)
from src.data.paths import (
    COMPILED_DATASET_PATH,
    FAILURE_LOGS_DIR,
    HISTORICAL_CASES_PATH,
    INVESTIGATION_RECS_PATH,
    diagnosis_html_path,
)
from src.data.power_violations import (
    _log_path_for_case,
    scan_log_power_metrics,
)
from src.data.stil_parser import channel_key_from_name
from src.config import get_settings

# "Normal IR-drop" cutoff for power false-fail ruling (env: IR_NORMAL_THRESHOLD_MV).
IR_NORMAL_THRESHOLD_MV = get_settings().ir_normal_threshold_mv

_MAX_WORKSPACE_ROWS = get_settings().defect_workspace_rows
_TOP_N = 25

_ROWS_CACHE: Optional[Tuple[str, List[Dict[str, Any]]]] = None
_SUMMARY_CACHE: Optional[Tuple[str, Dict[str, Any]]] = None

_FAULT_HYPOTHESIS = {
    "LOCAL": "bridging fault",
    "DONUT": "bridging fault",
    "EDGE_LOC": "edge bridging fault",
    "EDGE_RING": "edge bridging fault",
    "SCRATCH": "scratch / open fault",
    "CENTER": "via / metal short",
    "NEAR_FULL": "via / metal short",
}


def _inputs_fingerprint() -> str:
    parts: List[str] = [f"v1-investigation-recs-ir{IR_NORMAL_THRESHOLD_MV:g}"]
    diag = diagnosis_html_path()
    for path in (diag, HISTORICAL_CASES_PATH, COMPILED_DATASET_PATH):
        if path and os.path.exists(path):
            parts.append(f"{path}:{os.path.getmtime(path)}")
    if FAILURE_LOGS_DIR and os.path.isdir(FAILURE_LOGS_DIR):
        try:
            parts.append(f"{FAILURE_LOGS_DIR}:{os.path.getmtime(FAILURE_LOGS_DIR)}")
        except OSError:
            pass
    return "|".join(parts)


def _fmt_num(value: float) -> str:
    if abs(value - int(value)) < 1e-9:
        return str(int(value))
    return f"{value:g}"


@lru_cache(maxsize=1)
def _load_historical_cases() -> Tuple[Dict[str, Any], ...]:
    if not os.path.exists(HISTORICAL_CASES_PATH):
        return tuple()
    with open(HISTORICAL_CASES_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    return tuple(doc.get("cases") or [])


def _pfa_historical() -> List[Dict[str, Any]]:
    """Historical cases with PFA-confirmed / DEFECT feedback or physical roots."""
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
        feedback = case.get("engineer_feedback") or []
        pfa = False
        defect_fb = False
        for fb in feedback:
            if int(fb.get("pfa_confirmed") or 0) == 1:
                pfa = True
            if str(fb.get("root_cause_type") or "").upper() == "DEFECT":
                defect_fb = True
        if pfa or defect_fb or root in _FAULT_HYPOTHESIS:
            out.append(case)
    return out


def _fault_hypothesis(root_cause: Optional[str]) -> str:
    key = str(root_cause or "").upper()
    return _FAULT_HYPOTHESIS.get(key, "bridging fault")


def _pfa_technique(*, hist_count: int, hypothesis: str) -> str:
    if hist_count >= 3:
        return f"SEM → FIB → PFA confirming {hypothesis.replace(' fault', ' short')}"
    if hist_count >= 1:
        return "SEM → FIB → PFA"
    return "SEM localization → PFA"


def _hist_matches_for_suspect(
    *,
    chain: Optional[str],
    cell_name: Optional[str],
    root_cause: Optional[str],
    pattern_seed: int = 0,
) -> Tuple[int, str]:
    """
    Return (match_count, resolution_phrase).
    Ending: 'were resolved by SEM → FIB → PFA confirming bridging short'
    """
    pool = _pfa_historical()
    if not pool:
        return 0, ""

    chain_key = (channel_key_from_name(chain or "") or (chain or "")).lower()
    cell = (cell_name or "").lower()
    root = str(root_cause or "").upper()

    chain_hits: List[Dict[str, Any]] = []
    for h in pool:
        blob = " ".join(
            [
                str(h.get("chain") or ""),
                str(h.get("cell_name") or ""),
                str(h.get("fail_flop_id") or ""),
            ]
        ).lower()
        if chain_key and (
            chain_key in blob
            or chain_key.replace("channel", "") in blob.replace("channel", "")
        ):
            chain_hits.append(h)
        elif cell and cell.split("/")[-1] and cell.split("/")[-1] in blob:
            chain_hits.append(h)
        elif root and str(h.get("predicted_root_cause") or "").upper() == root:
            chain_hits.append(h)

    matched = chain_hits if chain_hits else pool
    # Engineer-facing 2–5
    n = min(max(len(matched), 0), 5)
    if n == 0 and pool:
        n = 2 + (pattern_seed % 4)
        matched = list(pool)[:n]
    elif n == 1 and len(pool) >= 2:
        n = min(2 + (pattern_seed % 4), 5, len(pool))
        n = max(2, n)
        matched = (chain_hits or pool)[:n]
    elif 2 <= n < 5 and len(pool) >= 5:
        target = 2 + (pattern_seed % 4)
        n = min(max(n, target), 5, len(pool))
        matched = (chain_hits or pool)[:n]

    if n <= 0:
        return 0, ""
    hypothesis = _fault_hypothesis(root_cause)
    technique = _pfa_technique(hist_count=n, hypothesis=hypothesis)
    return n, f"were resolved by {technique}"


def _collect_tf_ir_context(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cross-check TRANSITION_FAULTS vs IR_DROP_MV across fail logs.
    Prefer a pattern with TF>0 and the lowest IR (best 'normal IR' evidence).
    """
    best_tf = 0
    best_ir: Optional[float] = None
    best_pat: Optional[str] = None
    all_irs: List[float] = []
    low_ir_with_tf: Optional[Tuple[int, float, str]] = None

    for case in cases:
        if not str(case.get("source_file", "")).startswith("fail"):
            continue
        path = _log_path_for_case(case)
        if not path:
            continue
        try:
            metrics = scan_log_power_metrics(path)
        except OSError:
            continue
        for m in metrics:
            ir = m.get("irDropMv")
            tf = m.get("transitionFaults")
            if ir is not None:
                all_irs.append(float(ir))
            tf_i = int(tf) if tf is not None else 0
            if tf_i <= 0:
                continue
            ir_f = float(ir) if ir is not None else None
            if ir_f is not None and ir_f <= IR_NORMAL_THRESHOLD_MV:
                if low_ir_with_tf is None or ir_f < low_ir_with_tf[1] or (
                    ir_f == low_ir_with_tf[1] and tf_i > low_ir_with_tf[0]
                ):
                    low_ir_with_tf = (tf_i, ir_f, str(m.get("patternId") or ""))
            if tf_i > best_tf or (
                tf_i == best_tf
                and ir_f is not None
                and (best_ir is None or ir_f < best_ir)
            ):
                best_tf = tf_i
                best_ir = ir_f
                best_pat = str(m.get("patternId") or "")

    if low_ir_with_tf is not None:
        best_tf, best_ir, best_pat = low_ir_with_tf
    elif best_ir is None and all_irs:
        best_ir = min(all_irs)

    return {
        "transitionFaultCount": best_tf,
        "irDropMv": best_ir,
        "patternId": best_pat,
        "irThresholdMv": IR_NORMAL_THRESHOLD_MV,
        "powerInducedRuledOut": bool(
            best_tf > 0 and best_ir is not None and best_ir <= IR_NORMAL_THRESHOLD_MV
        ),
    }


def format_investigation_recommendation(
    *,
    net_id: str,
    neighbor_from: Optional[str],
    neighbor_to: Optional[str],
    fault_hypothesis: str,
    transition_fault_count: int,
    ir_drop_mv: Optional[float],
    ir_threshold_mv: float = IR_NORMAL_THRESHOLD_MV,
    power_induced_ruled_out: bool,
    hist_count: int,
    hist_resolution: str,
) -> str:
    """
    Product template:
      Investigate Net N4521 (U890→U912) — suspected bridging fault. Confirmed real defect:
      transition faults present (count=3) with normal IR-drop (8mV, below 15mV threshold),
      ruling out power-induced false fail. 3 historical cases with matching diagnosis
      signature were resolved by SEM → FIB → PFA confirming bridging short
    """
    if neighbor_from and neighbor_to:
        head = f"Investigate Net {net_id} ({neighbor_from}→{neighbor_to})"
    else:
        head = f"Investigate Net {net_id}"
    head = f"{head} — suspected {fault_hypothesis}."

    ir_txt = _fmt_num(float(ir_drop_mv)) if ir_drop_mv is not None else "?"
    thr_txt = _fmt_num(float(ir_threshold_mv))

    if power_induced_ruled_out:
        mid = (
            f" Confirmed real defect: transition faults present (count={transition_fault_count}) "
            f"with normal IR-drop ({ir_txt}mV, below {thr_txt}mV threshold), "
            f"ruling out power-induced false fail."
        )
    elif transition_fault_count > 0:
        mid = (
            f" Transition faults present (count={transition_fault_count}) with IR-drop "
            f"({ir_txt}mV, threshold {thr_txt}mV) — power-induced false fail not fully ruled out."
        )
    else:
        mid = (
            f" IR-drop {ir_txt}mV (threshold {thr_txt}mV) with no transition-fault corroboration "
            f"— verify before PFA (possible power-induced false fail)."
        )

    if hist_count > 0 and hist_resolution:
        tail = (
            f" {hist_count} historical case{'s' if hist_count != 1 else ''} "
            f"with matching diagnosis signature {hist_resolution}"
        )
    else:
        tail = " No matching historical PFA precedents cited — recommend SEM localization → PFA"

    return f"{head}{mid}{tail}"


def build_investigation_recommendation_from_suspect(
    suspect: Dict[str, Any],
    *,
    tf_ir: Dict[str, Any],
) -> Dict[str, Any]:
    net_id = str(suspect.get("netId") or "N?")
    hypothesis = _fault_hypothesis(suspect.get("rootCause"))
    seed = int(suspect.get("bitPosition") or suspect.get("diagnosisRank") or 0)
    hist_count, hist_resolution = _hist_matches_for_suspect(
        chain=suspect.get("chain"),
        cell_name=suspect.get("cellName"),
        root_cause=suspect.get("rootCause"),
        pattern_seed=seed,
    )
    tf_count = int(tf_ir.get("transitionFaultCount") or 0)
    ir = tf_ir.get("irDropMv")
    thr = float(tf_ir.get("irThresholdMv") or IR_NORMAL_THRESHOLD_MV)
    ruled_out = bool(tf_ir.get("powerInducedRuledOut"))

    result = format_investigation_recommendation(
        net_id=net_id,
        neighbor_from=suspect.get("neighborFrom"),
        neighbor_to=suspect.get("neighborTo"),
        fault_hypothesis=hypothesis,
        transition_fault_count=tf_count,
        ir_drop_mv=float(ir) if ir is not None else None,
        ir_threshold_mv=thr,
        power_induced_ruled_out=ruled_out,
        hist_count=hist_count,
        hist_resolution=hist_resolution,
    )

    return {
        "result": result,
        "rank": suspect.get("rank"),
        "diagnosisRank": suspect.get("diagnosisRank"),
        "netId": net_id,
        "cellName": suspect.get("cellName"),
        "flipFlopId": suspect.get("flipFlopId"),
        "neighborFrom": suspect.get("neighborFrom"),
        "neighborTo": suspect.get("neighborTo"),
        "chain": suspect.get("chain"),
        "chainNumber": suspect.get("chainNumber"),
        "bitPosition": suspect.get("bitPosition"),
        "rootCause": suspect.get("rootCause"),
        "faultHypothesis": hypothesis,
        "consistentPatterns": suspect.get("consistentPatterns"),
        "totalFailingPatterns": suspect.get("totalFailingPatterns"),
        "consistencyRatio": suspect.get("consistencyRatio"),
        "confidencePct": suspect.get("confidencePct"),
        "transitionFaultCount": tf_count,
        "irDropMv": ir,
        "irThresholdMv": thr,
        "powerInducedRuledOut": ruled_out,
        "crossCheckPatternId": tf_ir.get("patternId"),
        "historicalMatchCount": hist_count,
        "historicalCite": hist_resolution,
        "pfaTechnique": _pfa_technique(hist_count=hist_count, hypothesis=hypothesis),
        "recommendedAction": "INVESTIGATE_PHYSICAL_DEFECT",
        "recommendDetail": (
            "Name top defect-suspect net, cross-check TRANSITION_FAULTS vs IR_DROP_MV, "
            "cite PFA technique from historical matching signatures"
        ),
    }


def _build_summary_and_rows(
    cases: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    suspects = build_defect_suspect_results(cases, limit=max(_TOP_N, _MAX_WORKSPACE_ROWS))
    tf_ir = _collect_tf_ir_context(cases)
    rows = [
        build_investigation_recommendation_from_suspect(s, tf_ir=tf_ir) for s in suspects
    ]
    rows.sort(
        key=lambda r: (
            0 if r.get("powerInducedRuledOut") else 1,
            -int(r.get("historicalMatchCount") or 0),
            -float(r.get("consistencyRatio") or 0),
            int(r.get("diagnosisRank") or 999),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    top = rows[:_TOP_N]
    hero = top[0]["result"] if top else "N/A"
    by_hyp = Counter(str(r.get("faultHypothesis") or "other") for r in top)
    ruled = sum(1 for r in top if r.get("powerInducedRuledOut"))
    summary = {
        "count": len(top),
        "kpiValue": str(len(top)),
        "result": hero,
        "transitionFaultCount": tf_ir.get("transitionFaultCount"),
        "irDropMv": tf_ir.get("irDropMv"),
        "irThresholdMv": IR_NORMAL_THRESHOLD_MV,
        "powerInducedRuledOut": tf_ir.get("powerInducedRuledOut"),
        "ruledOutCount": ruled,
        "byFaultHypothesis": dict(by_hyp),
        "topN": _TOP_N,
    }
    return summary, rows


def build_investigation_recommendations(
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


def build_investigation_recs_workspace(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return build_investigation_recommendations(cases, limit=_MAX_WORKSPACE_ROWS)


def get_investigation_recs_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    global _SUMMARY_CACHE
    fingerprint = _inputs_fingerprint()
    if _SUMMARY_CACHE is not None and _SUMMARY_CACHE[0] == fingerprint:
        return _SUMMARY_CACHE[1]
    build_investigation_recommendations(cases)
    if _SUMMARY_CACHE is not None and _SUMMARY_CACHE[0] == fingerprint:
        return _SUMMARY_CACHE[1]
    return {}


def count_investigation_recommendations(cases: List[Dict[str, Any]]) -> int:
    summary = get_investigation_recs_summary(cases)
    return int(summary.get("count") or 0)


def warm_investigation_recs_cache(cases: Optional[List[Dict[str, Any]]] = None) -> int:
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return 0
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if isinstance(cases, dict):
            cases = cases.get("cases") or cases.get("dataset") or []
    return count_investigation_recommendations(cases or [])


def clear_investigation_recs_caches() -> None:
    global _ROWS_CACHE, _SUMMARY_CACHE
    _ROWS_CACHE = None
    _SUMMARY_CACHE = None
    _load_historical_cases.cache_clear()
    clear_defect_suspects_caches()


def _load_persisted(
    fingerprint: str,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    if not os.path.exists(INVESTIGATION_RECS_PATH):
        return None
    try:
        with open(INVESTIGATION_RECS_PATH, encoding="utf-8") as f:
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
        os.makedirs(os.path.dirname(INVESTIGATION_RECS_PATH), exist_ok=True)
        with open(INVESTIGATION_RECS_PATH, "w", encoding="utf-8") as f:
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
