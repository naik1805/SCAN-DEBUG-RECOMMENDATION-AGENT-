"""Parse Tessent/EDT STIL ScanStructures for scan-chain topology metadata."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, Optional

from src.data.paths import STIL_PATH


_CHAIN_BLOCK_RE = re.compile(
    r"ScanChain\s+([^\s{]+)\s*\{([^}]*)\}",
    re.MULTILINE | re.DOTALL,
)
_CHANNEL_RE = re.compile(r"channel(\d+)", re.IGNORECASE)


def _field(block: str, name: str) -> Optional[str]:
    m = re.search(rf'{name}\s+"?([^";\n]+)"?\s*;', block, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _int_field(block: str, name: str, default: int = 0) -> int:
    raw = _field(block, name)
    if raw is None:
        return default
    try:
        return int(re.sub(r"\D", "", raw) or default)
    except ValueError:
        return default


def channel_key_from_name(name: str) -> Optional[str]:
    """Normalize channel1 / CH1 / edt_block_channel12 → channel12."""
    if not name:
        return None
    m = _CHANNEL_RE.search(str(name))
    if not m:
        return None
    return f"channel{int(m.group(1))}"


def chain_display_number(name: str) -> Optional[int]:
    key = channel_key_from_name(name)
    if not key:
        return None
    return int(key.replace("channel", ""))


@lru_cache(maxsize=2)
def load_stil_scan_chains(stil_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Return map of channelN → {chain_id, channel, scan_length, scan_in, scan_out, clock}.
    Reads only the ScanStructures preamble (stops before PatternBurst / large vector data).
    """
    path = stil_path or STIL_PATH
    chains: Dict[str, Dict[str, Any]] = {}
    if not path or not os.path.exists(path):
        return chains

    # STIL pattern body is huge; ScanStructures sits near the top.
    chunk = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            piece = f.read(256_000)
            if not piece:
                break
            chunk += piece
            if "PatternBurst" in chunk or "PatternBlocks" in chunk or len(chunk) > 2_000_000:
                break

    structures = chunk
    m = re.search(r"ScanStructures\s*\{", chunk)
    if m:
        # Take a bounded window after ScanStructures start
        structures = chunk[m.start() : m.start() + 200_000]

    for match in _CHAIN_BLOCK_RE.finditer(structures):
        chain_id = match.group(1).strip()
        body = match.group(2)
        key = channel_key_from_name(chain_id)
        if not key:
            continue
        # Prefer core EDT channels over phy duplicates with same channel number
        if key in chains and "phy_inst" in chain_id and "core_inst" in chains[key].get("chain_id", ""):
            continue
        chains[key] = {
            "chain_id": chain_id,
            "channel": key,
            "chain_number": int(key.replace("channel", "")),
            "scan_length": _int_field(body, "ScanLength", 234),
            "scan_in": _field(body, "ScanIn") or "",
            "scan_out": _field(body, "ScanOut") or "",
            "clock": _field(body, "ScanMasterClock") or "",
        }
    return chains


def get_chain_meta(chain_name: str, stil_path: Optional[str] = None) -> Dict[str, Any]:
    chains = load_stil_scan_chains(stil_path)
    key = channel_key_from_name(chain_name) or ""
    return chains.get(key, {})
