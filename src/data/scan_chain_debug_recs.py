"""
Scan Chain Debug Recommendations KPI.

Template:
  chain #, cell name, fault type from diagnosis, plus closest historical match
  fix if similarity score is high.

Inputs:
  broken-chain isolation + diagnosis fault type + historical cases
  (similarity on chain-length / position signature)

Output example:
  Inspect Scan Chain 12 near cell SDFF_1045 — stuck-at-0 (diagnosis rank 1);
  3 historical cases with matching signature root-caused to scan-in net open
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from src.data.chain_break_diagnosis import display_cell_name, format_break_result
from src.data.log_fault_inference import infer_fault_type_from_log
from src.data.paths import (
    COMPILED_DATASET_PATH,
    FAILURE_LOGS_DIR,
    HISTORICAL_CASES_PATH,
    SCAN_CHAIN_DEBUG_RECS_PATH,
)
from src.data.stil_parser import chain_display_number
from src.config import get_settings

# Similarity on chain-length + position must be at least this to cite historical fix
HIGH_SIMILARITY = get_settings().scan_chain_high_similarity

_ROWS_CACHE: Optional[Tuple[str, List[Dict[str, Any]]]] = None


def _inputs_fingerprint() -> str:
    parts: List[str] = ["v1-scan-chain-debug-recs"]
    for path in (COMPILED_DATASET_PATH, HISTORICAL_CASES_PATH):
        if path and os.path.exists(path):
            parts.append(f"{path}:{os.path.getmtime(path)}")
    if FAILURE_LOGS_DIR and os.path.isdir(FAILURE_LOGS_DIR):
        try:
            parts.append(f"{FAILURE_LOGS_DIR}:{os.path.getmtime(FAILURE_LOGS_DIR)}")
        except OSError:
            pass
    return "|".join(parts)


def _load_persisted(fingerprint: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(SCAN_CHAIN_DEBUG_RECS_PATH):
        return None
    try:
        with open(SCAN_CHAIN_DEBUG_RECS_PATH, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("fingerprint") != fingerprint:
            return None
        rows = doc.get("rows")
        return rows if isinstance(rows, list) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _persist(fingerprint: str, rows: List[Dict[str, Any]]) -> None:
    try:
        os.makedirs(os.path.dirname(SCAN_CHAIN_DEBUG_RECS_PATH), exist_ok=True)
        with open(SCAN_CHAIN_DEBUG_RECS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "count": len(rows),
                    "rows": rows,
                },
                f,
                separators=(",", ":"),
            )
    except OSError:
        pass


def _parse_int(value: Any) -> Optional[int]:
    if value is None or value == "" or str(value).strip() in ("—", "-", "N/A"):
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else None


@lru_cache(maxsize=1)
def _load_historical_cases() -> Tuple[Dict[str, Any], ...]:
    if not os.path.exists(HISTORICAL_CASES_PATH):
        return tuple()
    with open(HISTORICAL_CASES_PATH, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return tuple(doc.get("cases") or [])


def _fault_type_from_case(case: Dict[str, Any], candidate_bit: Optional[int] = None) -> str:
    """Infer SA0/SA1 from failure log expected vs observed at diagnosis bit."""
    fault, _meta = infer_fault_type_from_log(case, candidate_bit=candidate_bit)
    return fault


def _position_fix_phrase(bit: Optional[int], scan_length: int) -> str:
    if bit is None or scan_length <= 1:
        return "scan path open"
    frac = bit / float(scan_length - 1)
    # candidate_bit aligns with cell index from SI in this dataset (ff[N])
    if frac <= 0.33:
        return "scan-in net open"
    if frac >= 0.67:
        return "scan-out net open"
    return "mid-chain net open"


def _root_cause_phrase(
    matched: List[Dict[str, Any]],
    bit: Optional[int],
    scan_length: int,
) -> str:
    causes = [
        str(m.get("predicted_root_cause") or "").upper()
        for m in matched
        if m.get("predicted_root_cause")
    ]
    causes = [c for c in causes if c and c not in ("NONE", "NULL")]
    top = Counter(causes).most_common(1)[0][0] if causes else ""
    pos = _position_fix_phrase(bit, scan_length)

    # Engineer-facing fix language when signature matches physical/defect history
    if top in ("DEFECT", "SCRATCH", "LOCAL", "RANDOM", "EDGE_LOC", "EDGE_RING", "DONUT"):
        return pos
    if top == "CENTER" or top == "NEAR_FULL":
        return f"capture IR-drop secondary fail ({pos})"
    if top == "NORMAL":
        return f"constraint-masked chain fail ({pos})"
    if top:
        return f"{top.lower().replace('_', ' ')} ({pos})"
    return pos


def signature_similarity(
    bit: Optional[int],
    scan_length: int,
    hist: Dict[str, Any],
) -> float:
    """Similarity on chain-length + position (offset_from_scan_in vs candidate bit)."""
    if bit is None:
        return 0.0
    h_len = _parse_int(hist.get("chain_length")) or scan_length
    h_off = _parse_int(hist.get("offset_from_scan_in"))
    if h_off is None:
        # Fall back to bracket index in cell name
        cell = str(hist.get("cell_name") or "")
        m = re.search(r"\[(\d+)\]", cell)
        h_off = int(m.group(1)) if m else None
    if h_off is None:
        return 0.0

    len_score = 1.0 if h_len == scan_length else max(0.0, 1.0 - abs(h_len - scan_length) / max(scan_length, 1))
    pos_delta = abs(h_off - bit)
    # Tight position window (±18 cells) for "matching signature"
    if pos_delta > 18:
        return 0.0
    pos_score = max(0.0, 1.0 - pos_delta / 18.0)
    return round(0.25 * len_score + 0.75 * pos_score, 4)


def find_historical_signature_matches(
    case: Dict[str, Any],
    *,
    min_similarity: float = HIGH_SIMILARITY,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], float]:
    """Return (all high matches, closest match, closest score)."""
    bit = _parse_int(case.get("candidate_bit") or case.get("exact_break_bit"))
    scan_length = (
        _parse_int(case.get("stil_scan_length"))
        or _parse_int(case.get("shift_cycles"))
        or 234
    )
    chain = str(case.get("chain_name") or "").lower()

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for hist in _load_historical_cases():
        score = signature_similarity(bit, scan_length, hist)
        # Small boost for same channel
        if chain and str(hist.get("chain") or "").lower() == chain:
            score = min(1.0, score + 0.08)
        if score >= min_similarity:
            scored.append((score, hist))

    scored.sort(key=lambda t: t[0], reverse=True)
    matches = [h for _, h in scored]
    if not scored:
        return [], None, 0.0
    best_score, best = scored[0]
    return matches, best, best_score


def format_scan_chain_debug_recommendation(
    case: Dict[str, Any],
    *,
    diagnosis_rank: int,
) -> Dict[str, Any]:
    """Build one Scan Chain Debug Recommendations row."""
    break_info = format_break_result(
        case.get("chain_name") or "",
        case.get("candidate_bit") or case.get("exact_break_bit"),
        scan_length=case.get("stil_scan_length") or case.get("shift_cycles"),
        candidate_cell=case.get("candidate_cell"),
        location_status=case.get("location_status"),
    )
    chain_num = break_info.get("chain_number") or chain_display_number(case.get("chain_name") or "")
    bit = break_info.get("candidate_bit")
    cell = break_info.get("cell_label") or display_cell_name(case.get("candidate_cell"), bit)
    fault, fault_meta = infer_fault_type_from_log(case, candidate_bit=bit)
    scan_length = break_info.get("scan_length") or 234

    matches, closest, best_score = find_historical_signature_matches(case)
    n_hist = len(matches)

    chain_label = f"Scan Chain {chain_num}" if chain_num is not None else "Scan Chain"
    head = f"Inspect {chain_label} near cell {cell} - {fault} (diagnosis rank {diagnosis_rank})"

    if n_hist > 0 and best_score >= HIGH_SIMILARITY:
        fix = _root_cause_phrase(matches, bit, int(scan_length))
        hist_part = (
            f"{n_hist} historical case{'s' if n_hist != 1 else ''} with matching signature "
            f"root-caused to {fix}"
        )
        result = f"{head}; {hist_part}"
    else:
        result = f"{head}; no high-similarity historical match"

    return {
        "lotId": case.get("lot_id"),
        "dieLabel": case.get("die_label"),
        "chain": break_info.get("chain"),
        "chainName": case.get("chain_name"),
        "chainNumber": chain_num,
        "cellLabel": cell,
        "candidateBit": bit,
        "faultType": fault,
        "faultSource": fault_meta.get("source"),
        "faultVotes": fault_meta.get("votes"),
        "diagnosisRank": diagnosis_rank,
        "historicalMatchCount": n_hist,
        "historicalSimilarity": round(best_score, 3),
        "historicalRootCause": _root_cause_phrase(matches, bit, int(scan_length)) if matches else "",
        "closestHistoricalCell": (closest or {}).get("cell_name") if closest else "",
        "scanLength": scan_length,
        "scanIn": break_info.get("scan_in") or "",
        "scanOut": break_info.get("scan_out") or "",
        "failCount": case.get("fail_count") or case.get("mismatch_count"),
        "result": result,
    }


def _build_rows(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    broken = [c for c in cases if c.get("has_break") and c.get("chain_name")]
    # Diagnosis rank 1 = highest fail evidence
    ordered = sorted(
        broken,
        key=lambda c: (
            -(c.get("fail_count") or c.get("mismatch_count") or 0),
            str(c.get("lot_id") or ""),
            str(c.get("die_label") or ""),
        ),
    )
    return [
        format_scan_chain_debug_recommendation(case, diagnosis_rank=rank)
        for rank, case in enumerate(ordered, start=1)
    ]


def build_scan_chain_debug_recommendations(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank broken-chain cases and emit Inspect Scan Chain recommendations."""
    global _ROWS_CACHE
    fingerprint = _inputs_fingerprint()
    if _ROWS_CACHE is not None and _ROWS_CACHE[0] == fingerprint:
        return _ROWS_CACHE[1]

    persisted = _load_persisted(fingerprint)
    if persisted is not None:
        _ROWS_CACHE = (fingerprint, persisted)
        return persisted

    rows = _build_rows(cases)
    _persist(fingerprint, rows)
    _ROWS_CACHE = (fingerprint, rows)
    return rows


def count_scan_chain_debug_recommendations(cases: List[Dict[str, Any]]) -> int:
    return len(build_scan_chain_debug_recommendations(cases))


def warm_scan_chain_debug_recs_cache(cases: Optional[List[Dict[str, Any]]] = None) -> int:
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return 0
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if isinstance(cases, dict):
            cases = cases.get("cases") or cases.get("dataset") or []
    return len(build_scan_chain_debug_recommendations(cases or []))


def clear_scan_chain_debug_recs_caches() -> None:
    global _ROWS_CACHE
    _ROWS_CACHE = None
    _load_historical_cases.cache_clear()
