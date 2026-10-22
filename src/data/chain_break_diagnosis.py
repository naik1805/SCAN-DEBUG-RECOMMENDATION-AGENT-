"""
Scan-chain break localization for Broken Chains Detected KPI.

How to implement:
  Run shift-only patterns from failing bitmap; find first bit position where
  expected≠observed consistently across patterns → maps to chain # + cell
  index via STIL chain order

Output example:
  Chain 12, break isolated at bit position 45 (cell SDFF_1045)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.data.stil_parser import chain_display_number, get_chain_meta

HOW_TO_IMPLEMENT = (
    "Run shift-only patterns from failing bitmap; find first bit position where "
    "expected≠observed consistently across patterns → maps to chain # + cell "
    "index via STIL chain order"
)


def _parse_bit(value: Any) -> Optional[int]:
    if value is None or value == "" or str(value).strip() in ("—", "-", "N/A"):
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else None


def display_cell_name(candidate_cell: Optional[str], bit: Optional[int]) -> str:
    """Map diagnosis cell / bit to a compact SDFF_* label for UI."""
    cell = (candidate_cell or "").strip()
    if cell and "LOCATION_UNCERTAIN" not in cell.upper() and cell not in ("—", "-"):
        bracket = re.search(r"\[(\d+)\]", cell)
        if bracket:
            return f"SDFF_{bracket.group(1)}"
        ff = re.search(r"(?:SDFF|FF)_?(\d+)", cell, re.IGNORECASE)
        if ff:
            return f"SDFF_{ff.group(1)}"
        # last path segment if hierarchical
        tail = cell.split("/")[-1]
        if tail:
            return tail
    if bit is not None:
        return f"SDFF_{bit}"
    return "UNKNOWN"


def format_break_result(
    chain_name: str,
    candidate_bit: Any,
    *,
    scan_length: Optional[int] = None,
    candidate_cell: Optional[str] = None,
    location_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Build UI-ready break isolation result for one die/chain."""
    meta = get_chain_meta(chain_name)
    length = scan_length or meta.get("scan_length") or 234
    chain_num = chain_display_number(chain_name) or meta.get("chain_number")
    bit = _parse_bit(candidate_bit)
    if bit is not None:
        bit = max(0, min(bit, max(0, int(length) - 1)))

    if chain_num is None:
        chain_label = str(chain_name or "unknown")
    else:
        chain_label = f"Chain {chain_num}"

    cell_label = display_cell_name(candidate_cell, bit)
    if bit is None:
        result = f"{chain_label}, break location uncertain (no consensus fail bit)"
    else:
        result = (
            f"{chain_label}, break isolated at bit position {bit} (cell {cell_label})"
        )

    status = (location_status or "").upper() or "UNCERTAIN"
    return {
        "chain": chain_label,
        "chain_name": chain_name,
        "chain_number": chain_num,
        "result": result,
        "candidate_bit": bit,
        "cell_label": cell_label,
        "scan_length": int(length),
        "scan_in": meta.get("scan_in") or "",
        "scan_out": meta.get("scan_out") or "",
        "stil_clock": meta.get("clock") or "",
        "candidate_cell": candidate_cell or "",
        "location_status": status,
        "how_to_implement": HOW_TO_IMPLEMENT,
    }


def enrich_case_with_break_isolation(case: Dict[str, Any]) -> Dict[str, Any]:
    """Attach break isolation fields onto a compiled die case (in-place + return)."""
    if not case.get("has_break"):
        return case
    chain = case.get("chain_name") or ""
    if not chain:
        return case

    shift_cycles = case.get("shift_cycles")
    try:
        scan_length = int(shift_cycles) if shift_cycles not in (None, "") else None
    except (TypeError, ValueError):
        scan_length = None

    info = format_break_result(
        chain,
        case.get("candidate_bit") or case.get("exact_break_bit"),
        scan_length=scan_length,
        candidate_cell=case.get("candidate_cell"),
        location_status=case.get("location_status"),
    )
    case["break_isolation_result"] = info["result"]
    case["break_cell_label"] = info["cell_label"]
    case["chain_display"] = info["chain"]
    case["stil_scan_in"] = info["scan_in"]
    case["stil_scan_out"] = info["scan_out"]
    case["stil_scan_length"] = info["scan_length"]
    return case


def build_broken_chain_diagnosis_results(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ordered diagnosis results for Broken Chains Detected drill-down (all breaks)."""
    rows: List[Dict[str, Any]] = []
    for c in cases:
        if not c.get("has_break"):
            continue
        chain = c.get("chain_name") or ""
        if not chain:
            continue
        info = format_break_result(
            chain,
            c.get("candidate_bit") or c.get("exact_break_bit"),
            scan_length=c.get("stil_scan_length") or c.get("shift_cycles"),
            candidate_cell=c.get("candidate_cell"),
            location_status=c.get("location_status"),
        )
        rows.append(
            {
                "lotId": c.get("lot_id"),
                "dieLabel": c.get("die_label"),
                "chain": info["chain"],
                "chainName": info["chain_name"],
                "result": info["result"],
                "candidateBit": info["candidate_bit"],
                "cellLabel": info["cell_label"],
                "candidateCell": info["candidate_cell"],
                "locationStatus": info["location_status"],
                "scanIn": info["scan_in"],
                "scanOut": info["scan_out"],
                "scanLength": info["scan_length"],
                "failCount": c.get("fail_count") or c.get("mismatch_count"),
                "howToImplement": HOW_TO_IMPLEMENT,
            }
        )
    rows.sort(key=lambda r: (r.get("lotId") or "", r.get("dieLabel") or "", r.get("chain") or ""))
    return rows
