"""
Defect Suspects KPI (Physical Defect Investigation).

Inputs:
  Diagnosis FR-002 ranked candidates (Exact Cell Name, corroborating/observations)
  + failing-bitmap / pattern consistency (corroborating ÷ observations)
  + STIL (chain number / scan-order neighbor resolution for Ua→Ub)

Implementation:
  Pull top-N ranked candidates from diagnosis; validate each by the fraction of
  failing patterns consistent with that candidate being the true fault site.

Output:
  Net N4521 (U890→U912) — diagnosis rank 1, consistent with 11/12 failing patterns;
  Net N4487 — rank 2, consistent with 6/12
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
from src.data.paths import (
    COMPILED_DATASET_PATH,
    DEFECT_SUSPECTS_PATH,
    FAILING_BITMAPS_PATH,
    STIL_PATH,
    diagnosis_html_path,
)
from src.data.stil_parser import chain_display_number, get_chain_meta

# Physical-defect style root causes (aligned with action_inference).
_PHYSICAL_ROOTS = frozenset(
    {"LOCAL", "DONUT", "EDGE_RING", "EDGE_LOC", "CENTER", "NEAR_FULL"}
)

_TOP_N = 25
_MAX_WORKSPACE_ROWS = 40

_ROWS_CACHE: Optional[Tuple[str, List[Dict[str, Any]]]] = None
_SUMMARY_CACHE: Optional[Tuple[str, Dict[str, Any]]] = None


def _inputs_fingerprint() -> str:
    parts: List[str] = [f"v1-defect-suspects-top{_TOP_N}"]
    diag = diagnosis_html_path()
    for path in (diag, FAILING_BITMAPS_PATH, STIL_PATH, COMPILED_DATASET_PATH):
        if path and os.path.exists(path):
            parts.append(f"{path}:{os.path.getmtime(path)}")
    return "|".join(parts)


def _parse_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else None


def _compact_u_label(cell_name: Optional[str], bit: Optional[int], ff_id: Optional[str] = None) -> str:
    """Product-style instance label: U890."""
    if ff_id:
        m = re.search(r"(\d+)", str(ff_id))
        if m:
            return f"U{m.group(1)}"
    if cell_name:
        bracket = re.search(r"\[(\d+)\]", str(cell_name))
        if bracket:
            return f"U{bracket.group(1)}"
        m = re.search(r"(?:SDFF|FF)_?(\d+)", str(cell_name), re.IGNORECASE)
        if m:
            return f"U{m.group(1)}"
    if bit is not None:
        return f"U{int(bit)}"
    return "U?"


def _net_id_from_candidate(*, ff_id: Optional[str], bit: Optional[int], cell_name: Optional[str]) -> str:
    """Synthesize Net N#### when diagnosis/STIL has no explicit netlist net."""
    if ff_id:
        m = re.search(r"(\d+)", str(ff_id))
        if m:
            return f"N{m.group(1)}"
    if bit is not None:
        return f"N{int(bit)}"
    if cell_name:
        digits = "".join(ch for ch in str(cell_name) if ch.isdigit())
        if digits:
            return f"N{digits[-4:]}"
    return "N?"


@lru_cache(maxsize=1)
def _load_fr002_candidates() -> Tuple[Dict[str, Any], ...]:
    """Full FR-002 suspected failing cells with cell/net resolution fields."""
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
        table = _find_table_by_header(parser.tables, "corroborating patterns")
        if not table:
            return tuple()
        idx = _header_index(table[0])
        rows: List[Dict[str, Any]] = []
        for row in table[1:]:
            chain = _row_get(row, idx, "scan", "chain")
            if not chain:
                continue
            bit = _parse_int(_row_get(row, idx, "bit", "position"))
            cell = (_row_get(row, idx, "exact", "cell") or "").strip() or None
            ff_id = (_row_get(row, idx, "flip", "flop") or "").strip() or None
            root = (_row_get(row, idx, "predicted", "root") or "").strip().upper() or "UNKNOWN"
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
            # Pattern consistency: fraction of failing patterns consistent with candidate
            total = max(observations, corroborating, 1)
            consistent = min(max(0, corroborating), total)
            rows.append(
                {
                    "chain": chain,
                    "bit": bit,
                    "cellName": cell,
                    "flipFlopId": ff_id,
                    "rootCause": root,
                    "confidencePct": max(0.0, confidence),
                    "corroborating": consistent,
                    "observations": total,
                    "consistencyRatio": consistent / total,
                }
            )
        return tuple(rows)
    except Exception:
        return tuple()


def _neighbor_labels(chain: str, bit: Optional[int], cell_name: Optional[str], ff_id: Optional[str]) -> Tuple[str, str]:
    """
    Resolve (Ua→Ub) along STIL scan order.
    Prefer bit±1 instance labels; clamp to chain ScanLength when known.
    """
    meta = get_chain_meta(chain) or {}
    length = int(meta.get("scan_length") or 234)
    center = bit if bit is not None else _parse_int(ff_id)
    if center is None:
        u = _compact_u_label(cell_name, bit, ff_id)
        return u, u
    lo = max(0, center - 1)
    hi = min(length - 1, center + 1)
    if hi == lo:
        hi = min(length - 1, lo + 1)
    return f"U{lo}", f"U{hi}"


def format_defect_suspect_clause(
    *,
    net_id: str,
    diagnosis_rank: int,
    consistent: int,
    total: int,
    neighbor_from: Optional[str] = None,
    neighbor_to: Optional[str] = None,
    include_neighbors: bool = True,
) -> str:
    """
    One clause:
      Net N4521 (U890→U912) — diagnosis rank 1, consistent with 11/12 failing patterns
      Net N4487 — rank 2, consistent with 6/12
    """
    if include_neighbors and neighbor_from and neighbor_to and diagnosis_rank == 1:
        head = f"Net {net_id} ({neighbor_from}→{neighbor_to})"
        rank_txt = f"diagnosis rank {diagnosis_rank}"
        patterns_txt = f"{consistent}/{total} failing patterns"
    else:
        head = f"Net {net_id}"
        rank_txt = f"rank {diagnosis_rank}"
        patterns_txt = f"{consistent}/{total}"
    return f"{head} — {rank_txt}, consistent with {patterns_txt}"


def format_defect_suspects_recommendation(clauses: List[str]) -> str:
    """Join top clauses with '; ' (product multi-suspect line)."""
    return "; ".join(c for c in clauses if c)


def format_defect_suspects_kpi_value(count: int) -> str:
    return str(int(count))


def _build_summary_and_rows(
    cases: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    _ = cases  # fingerprint already covers compiled dataset; FR-002 is primary input
    candidates = [
        c
        for c in _load_fr002_candidates()
        if str(c.get("rootCause") or "").upper() in _PHYSICAL_ROOTS
    ]
    if not candidates:
        # Fall back to all FR-002 if physical filter is empty
        candidates = list(_load_fr002_candidates())

    candidates.sort(
        key=lambda c: (
            -float(c.get("confidencePct") or 0),
            -float(c.get("consistencyRatio") or 0),
            -int(c.get("corroborating") or 0),
            str(c.get("chain") or ""),
            int(c.get("bit") or 0),
        )
    )
    # Dedupe by net id (keep highest confidence)
    seen_nets: set[str] = set()
    ranked: List[Dict[str, Any]] = []
    for c in candidates:
        net_id = _net_id_from_candidate(
            ff_id=c.get("flipFlopId"),
            bit=c.get("bit"),
            cell_name=c.get("cellName"),
        )
        if net_id in seen_nets:
            continue
        seen_nets.add(net_id)
        ranked.append(c)
        if len(ranked) >= max(_TOP_N * 2, _MAX_WORKSPACE_ROWS):
            break

    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(ranked[: max(_TOP_N, _MAX_WORKSPACE_ROWS)], start=1):
        bit = c.get("bit")
        chain = str(c.get("chain") or "")
        net_id = _net_id_from_candidate(
            ff_id=c.get("flipFlopId"),
            bit=bit,
            cell_name=c.get("cellName"),
        )
        u_from, u_to = _neighbor_labels(chain, bit, c.get("cellName"), c.get("flipFlopId"))
        consistent = int(c.get("corroborating") or 0)
        total = int(c.get("observations") or 1)
        clause = format_defect_suspect_clause(
            net_id=net_id,
            diagnosis_rank=i,
            consistent=consistent,
            total=total,
            neighbor_from=u_from,
            neighbor_to=u_to,
            include_neighbors=True,
        )
        chain_num = chain_display_number(chain)
        stil_meta = get_chain_meta(chain) or {}
        rows.append(
            {
                "result": clause,
                "rank": i,
                "diagnosisRank": i,
                "netId": net_id,
                "cellName": c.get("cellName"),
                "flipFlopId": c.get("flipFlopId"),
                "neighborFrom": u_from,
                "neighborTo": u_to,
                "chain": chain,
                "chainNumber": chain_num,
                "bitPosition": bit,
                "consistentPatterns": consistent,
                "totalFailingPatterns": total,
                "consistencyRatio": round(float(c.get("consistencyRatio") or 0), 4),
                "confidencePct": c.get("confidencePct"),
                "rootCause": c.get("rootCause"),
                "scanIn": stil_meta.get("scan_in"),
                "scanOut": stil_meta.get("scan_out"),
                "recommendedAction": "INVESTIGATE_PHYSICAL_DEFECT",
                "recommendDetail": (
                    f"Top diagnosis candidate {net_id}; pattern consistency "
                    f"{consistent}/{total} via failing-bitmap corroboration"
                ),
            }
        )

    top_for_kpi = rows[:_TOP_N]
    hero_clauses = [
        format_defect_suspect_clause(
            net_id=str(r["netId"]),
            diagnosis_rank=int(r["diagnosisRank"]),
            consistent=int(r["consistentPatterns"]),
            total=int(r["totalFailingPatterns"]),
            neighbor_from=r.get("neighborFrom"),
            neighbor_to=r.get("neighborTo"),
            include_neighbors=True,
        )
        for r in top_for_kpi[:2]
    ]
    # Product shows full pattern text on rank 1; shorten rank 2+
    if len(top_for_kpi) >= 2:
        r2 = top_for_kpi[1]
        hero_clauses[1] = format_defect_suspect_clause(
            net_id=str(r2["netId"]),
            diagnosis_rank=int(r2["diagnosisRank"]),
            consistent=int(r2["consistentPatterns"]),
            total=int(r2["totalFailingPatterns"]),
            include_neighbors=False,
        )

    recommendation = format_defect_suspects_recommendation(hero_clauses)
    by_root = Counter(str(r.get("rootCause") or "UNKNOWN") for r in top_for_kpi)
    summary = {
        "count": len(top_for_kpi),
        "totalCandidates": len(ranked),
        "kpiValue": format_defect_suspects_kpi_value(len(top_for_kpi)),
        "result": recommendation,
        "topNetId": top_for_kpi[0]["netId"] if top_for_kpi else None,
        "topConsistency": (
            f"{top_for_kpi[0]['consistentPatterns']}/{top_for_kpi[0]['totalFailingPatterns']}"
            if top_for_kpi
            else None
        ),
        "byRootCause": dict(by_root),
        "topN": _TOP_N,
    }
    return summary, rows


def build_defect_suspect_results(
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


def build_defect_suspect_workspace(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return build_defect_suspect_results(cases, limit=_MAX_WORKSPACE_ROWS)


def get_defect_suspect_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    global _SUMMARY_CACHE
    fingerprint = _inputs_fingerprint()
    if _SUMMARY_CACHE is not None and _SUMMARY_CACHE[0] == fingerprint:
        return _SUMMARY_CACHE[1]
    build_defect_suspect_results(cases)
    if _SUMMARY_CACHE is not None and _SUMMARY_CACHE[0] == fingerprint:
        return _SUMMARY_CACHE[1]
    return {}


def count_defect_suspects(cases: List[Dict[str, Any]]) -> int:
    summary = get_defect_suspect_summary(cases)
    return int(summary.get("count") or 0)


def defect_suspects_kpi_value(cases: List[Dict[str, Any]]) -> str:
    summary = get_defect_suspect_summary(cases)
    return str(summary.get("kpiValue") or "0")


def warm_defect_suspects_cache(cases: Optional[List[Dict[str, Any]]] = None) -> int:
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return 0
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if isinstance(cases, dict):
            cases = cases.get("cases") or cases.get("dataset") or []
    return count_defect_suspects(cases or [])


def clear_defect_suspects_caches() -> None:
    global _ROWS_CACHE, _SUMMARY_CACHE
    _ROWS_CACHE = None
    _SUMMARY_CACHE = None
    _load_fr002_candidates.cache_clear()


def _load_persisted(
    fingerprint: str,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    if not os.path.exists(DEFECT_SUSPECTS_PATH):
        return None
    try:
        with open(DEFECT_SUSPECTS_PATH, encoding="utf-8") as f:
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
        os.makedirs(os.path.dirname(DEFECT_SUSPECTS_PATH), exist_ok=True)
        with open(DEFECT_SUSPECTS_PATH, "w", encoding="utf-8") as f:
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
