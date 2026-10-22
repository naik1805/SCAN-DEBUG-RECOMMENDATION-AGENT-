"""Parse STIL procedures for held pins and per-pattern active scan fan-out cones."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set

from src.data.paths import STIL_PATH

_PI_PIN_ORDER = [
    "XI",
    "TEST",
    "RESETN",
    "OLDI_VREF",
    "PVT_VM_LO",
    "PVT_VM_HI0",
    "PVT_VM_HI1",
    "PVT_VM_HI2",
]

_PATTERN_NUM_RE = re.compile(r"pattern_numer:(\d+)")
_PI_VALUE_RE = re.compile(r"_pi_=([01X]+)")
_PROCEDURE_RE = re.compile(r"procedure\s*=\s*([A-Za-z0-9_]+)")
_SCAN_IN_ANN_RE = re.compile(r"scan_in:([A-Za-z0-9_]+)")
_MACRO_CH_RE = re.compile(
    r"edt_block_channel\d+_([A-Za-z0-9_]+)_\"\s*=\s*([01X]+)",
)


def _display_pin_name(pin: str) -> str:
    if pin == "RESETN":
        return "RESET_N"
    return pin


def format_held_pins(pi_bits: str) -> str:
    """Format _pi_ vector as RESET_N=1,TEST=1,... for held constants."""
    parts: List[str] = []
    for idx, pin in enumerate(_PI_PIN_ORDER):
        if idx >= len(pi_bits):
            break
        bit = pi_bits[idx]
        if bit in ("0", "1"):
            parts.append(f"{_display_pin_name(pin)}={bit}")
    return ",".join(parts)


def _active_fanouts_from_macro(block: str) -> Set[str]:
    """Scan chains with non-masked shift data on the scan-in side (fan-out exercised)."""
    active: Set[str] = set()
    for signal, data in _MACRO_CH_RE.findall(block):
        payload = data[4:] if data.startswith("XXXX") else data
        if any(ch in "01" for ch in payload[:120]):
            active.add(signal)
    return active


@lru_cache(maxsize=2)
def load_stil_pattern_catalog(stil_path: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """
    Return pattern index → {held_pins, procedure, fanouts:set[str], pattern_label}.
    Reads Macro blocks per pattern to determine which scan-in fan-out cones are active.
    """
    path = stil_path or STIL_PATH
    catalog: Dict[int, Dict[str, Any]] = {}
    if not path or not os.path.exists(path):
        return catalog

    current_pat: Optional[int] = None
    in_macro = False
    macro_buf: List[str] = []

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            pat_match = _PATTERN_NUM_RE.search(line)
            if pat_match:
                current_pat = int(pat_match.group(1))
                catalog.setdefault(
                    current_pat,
                    {
                        "held_pins": "",
                        "procedure": "scan_shift_capture",
                        "fanouts": set(),
                        "pattern_label": f"P{current_pat + 1}",
                    },
                )

            if current_pat is None:
                continue

            proc_match = _PROCEDURE_RE.search(line)
            if proc_match:
                catalog[current_pat]["procedure"] = proc_match.group(1)

            pi_match = _PI_VALUE_RE.search(line)
            if pi_match and not catalog[current_pat]["held_pins"]:
                catalog[current_pat]["held_pins"] = format_held_pins(pi_match.group(1))

            for scan_in in _SCAN_IN_ANN_RE.findall(line):
                catalog[current_pat]["fanouts"].add(scan_in)

            if 'Macro "scan_edt_g1"' in line:
                in_macro = True
                macro_buf = [line]
                continue

            if in_macro:
                macro_buf.append(line)
                if line.strip() == "}":
                    catalog[current_pat]["fanouts"].update(
                        _active_fanouts_from_macro("".join(macro_buf))
                    )
                    in_macro = False

    return catalog


def clear_stil_pattern_catalog_cache() -> None:
    load_stil_pattern_catalog.cache_clear()
