"""
Scan Chain Confidence — weighted score from failure log patterns, diagnosis ambiguity,
and historical signature match.

Weighted score:
  pattern_consistency (0–1) × (1 / ambiguity_group) × historical_similarity_boost

Inputs:
  - corroborating / observation shift patterns from diagnosis FR-002 (preferred)
  - diagnosis localization ambiguity group (competing high-confidence cells near candidate)
  - historical case match strength (count × similarity)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from src.data.dataset_builder import (
    _TableCollector,
    _find_table_by_header,
    _header_index,
    _row_get,
)
from src.data.log_fault_inference import (
    _HL_TO_BIT,
    _log_path_for_case,
    parse_failing_waveform_pairs,
)
from src.data.paths import diagnosis_html_path
from src.data.scan_chain_debug_recs import find_historical_signature_matches

SOFT_AGREEMENT_WINDOW = 5
MIN_COMPETITOR_CONF_PCT = 50.0
MAX_CONFIDENCE_SCORE = 0.99


def _safe_pattern_ratio(consistent: int, total: int) -> Tuple[int, int, float]:
    """Clamp counts and return a safe (consistent, total, ratio) tuple."""
    safe_total = max(int(total), 1)
    safe_consistent = max(0, min(int(consistent), safe_total))
    return safe_consistent, safe_total, safe_consistent / safe_total


@lru_cache(maxsize=1)
def _load_fr002_index() -> List[Dict[str, Any]]:
    """Suspected failing cells (FR-002) indexed for corroboration + ambiguity."""
    path = diagnosis_html_path()
    if not path:
        return []

    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except OSError:
        return []

    try:
        parser = _TableCollector()
        parser.feed(html)
        table = _find_table_by_header(parser.tables, "corroborating patterns")
        if not table:
            return []

        idx = _header_index(table[0])
        rows: List[Dict[str, Any]] = []
        for row in table[1:]:
            chain = _row_get(row, idx, "scan", "chain")
            if not chain:
                continue
            try:
                bit = int(_row_get(row, idx, "bit", "position"))
            except (TypeError, ValueError):
                continue
            conf_raw = _row_get(row, idx, "confidence")
            try:
                confidence = float(str(conf_raw).replace("%", "").strip())
            except (TypeError, ValueError):
                confidence = 0.0
            try:
                corroborating = int(_row_get(row, idx, "corroborating") or 0)
                observations = int(_row_get(row, idx, "observations") or 0)
            except (TypeError, ValueError):
                corroborating, observations = 0, 0
            rows.append(
                {
                    "chain": chain,
                    "bit": bit,
                    "confidence": max(0.0, confidence),
                    "corroborating": max(0, corroborating),
                    "observations": max(0, observations),
                }
            )
        return rows
    except Exception:
        return []


def clear_fr002_cache() -> None:
    """Clear cached FR-002 index (for tests or dataset refresh)."""
    _load_fr002_index.cache_clear()


def _parse_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _fr002_match(case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Best FR-002 localization row for this chain + candidate bit (±5 soft window)."""
    chain = case.get("chain_name") or ""
    candidate_bit = _parse_int(case.get("candidate_bit"))
    if not chain or candidate_bit is None:
        return None

    best: Optional[Dict[str, Any]] = None
    best_delta = SOFT_AGREEMENT_WINDOW + 1
    for row in _load_fr002_index():
        if row["chain"] != chain:
            continue
        delta = abs(row["bit"] - candidate_bit)
        if delta > SOFT_AGREEMENT_WINDOW:
            continue
        if delta < best_delta or (delta == best_delta and row["confidence"] > (best or {}).get("confidence", -1)):
            best = row
            best_delta = delta
    return best


def _first_mismatch_bit(expected: List[str], actual: List[str]) -> Optional[int]:
    n = min(len(expected), len(actual))
    for i in range(n):
        exp = _HL_TO_BIT.get(expected[i].upper())
        act = _HL_TO_BIT.get(actual[i].upper())
        if exp and act and exp != act:
            return i
    return None


def pattern_consistency_stats(case: Dict[str, Any]) -> Tuple[int, int, float]:
    """
    Return (consistent_patterns, total_shift_patterns, consistency_ratio).
    Prefer diagnosis FR-002 corroborating/observation counts for the localized cell.
    """
    fr_match = _fr002_match(case)
    if fr_match and fr_match["observations"] > 0:
        return _safe_pattern_ratio(fr_match["corroborating"], fr_match["observations"])

    candidate_bit = _parse_int(case.get("candidate_bit"))
    fail_count = int(case.get("fail_count") or case.get("mismatch_count") or 0)
    log_path = _log_path_for_case(case)
    chain = case.get("chain_name") or ""

    if not log_path or not chain or candidate_bit is None:
        total = max(fail_count, 1)
        soft_pct = max(0.0, float(case.get("break_confidence_pct") or 0))
        consistent = max(0, round(soft_pct * total))
        if consistent == 0 and soft_pct > 0:
            consistent = 1
        return _safe_pattern_ratio(consistent, total)

    pairs = parse_failing_waveform_pairs(log_path, chain)
    mismatch_bits: List[int] = []
    for exp, act in pairs:
        bit = _first_mismatch_bit(exp, act)
        if bit is not None:
            mismatch_bits.append(bit)

    total = len(mismatch_bits) or max(fail_count, 1)
    consistent = sum(1 for bit in mismatch_bits if abs(bit - candidate_bit) <= SOFT_AGREEMENT_WINDOW)

    if not mismatch_bits and fail_count:
        soft_pct = max(0.0, float(case.get("break_confidence_pct") or 0))
        consistent = max(0, round(soft_pct * fail_count))
        return _safe_pattern_ratio(consistent, fail_count)

    return _safe_pattern_ratio(consistent, total)


def ambiguity_group_size(case: Dict[str, Any]) -> int:
    """
    Diagnosis localization ambiguity: competing high-confidence suspected cells
    on the same chain within ±5 bits of the candidate break.
    """
    chain = case.get("chain_name") or ""
    candidate_bit = _parse_int(case.get("candidate_bit"))
    if not chain or candidate_bit is None:
        return max(1, int(case.get("cell_count") or 1))

    near_bits = {
        row["bit"]
        for row in _load_fr002_index()
        if row["chain"] == chain
        and abs(row["bit"] - candidate_bit) <= SOFT_AGREEMENT_WINDOW
        and row["confidence"] >= MIN_COMPETITOR_CONF_PCT
    }
    if near_bits:
        return max(1, len(near_bits))

    return max(1, int(case.get("cell_count") or 1))


def historical_similarity_boost(case: Dict[str, Any]) -> Tuple[float, int, float]:
    """Return (boost multiplier, similar historical case count, best similarity)."""
    matches, _, best_score = find_historical_signature_matches(case)
    count = len(matches)
    if count == 0:
        return 1.0, 0, 0.0
    boost = 1.0 + (count * best_score) / 3.0
    return min(2.5, boost), count, best_score


def compute_scan_chain_confidence(case: Dict[str, Any]) -> Dict[str, Any]:
    """Compute weighted confidence and engineer-facing breakdown."""
    consistent, total, pattern_ratio = pattern_consistency_stats(case)
    ambiguity = ambiguity_group_size(case)
    hist_boost, hist_count, hist_similarity = historical_similarity_boost(case)

    score = min(MAX_CONFIDENCE_SCORE, max(0.0, pattern_ratio * (1.0 / ambiguity) * hist_boost))
    pct = round(score * 100)

    result = (
        f"{pct}% ({consistent}/{total} patterns consistent, "
        f"ambiguity={ambiguity}, {hist_count} similar historical case"
        f"{'' if hist_count == 1 else 's'})"
    )

    return {
        "confidencePct": pct,
        "confidenceScore": round(score, 4),
        "patternConsistent": consistent,
        "patternTotal": total,
        "patternConsistency": round(pattern_ratio, 4),
        "ambiguityGroup": ambiguity,
        "historicalMatchCount": hist_count,
        "historicalSimilarity": round(hist_similarity, 3),
        "historicalBoost": round(hist_boost, 3),
        "result": result,
        "lotId": case.get("lot_id"),
        "dieLabel": case.get("die_label"),
        "chain": case.get("chain_display") or case.get("chain_name"),
        "chainName": case.get("chain_name"),
        "cellLabel": case.get("break_cell_label") or case.get("candidate_cell"),
        "candidateBit": _parse_int(case.get("candidate_bit")),
    }


def average_scan_chain_confidence(cases: List[Dict[str, Any]]) -> str:
    if not cases:
        return "0%"
    scores = [compute_scan_chain_confidence(c)["confidenceScore"] for c in cases]
    return f"{round(sum(scores) / len(scores) * 100)}%"


def build_scan_chain_confidence_results(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One confidence row per case, highest score first."""
    rows = [compute_scan_chain_confidence(c) for c in cases if c.get("has_break") or c.get("chain_name")]
    rows.sort(
        key=lambda r: (
            -r["confidenceScore"],
            -r["patternConsistent"],
            str(r.get("lotId") or ""),
        ),
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
