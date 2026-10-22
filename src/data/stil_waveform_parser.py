"""Parse STIL Timing / WaveformTable blocks for period and launch/capture edge deltas."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from src.data.paths import STIL_PATH

_PERIOD_RE = re.compile(r"Period\s+'(\d+(?:\.\d+)?)(ps|ns|us)'", re.IGNORECASE)
_EDGE_PS_RE = re.compile(r"'(\d+(?:\.\d+)?)(ps|ns|us)'", re.IGNORECASE)


def _to_ps(value: float, unit: str) -> float:
    u = unit.lower()
    if u == "ps":
        return value
    if u == "ns":
        return value * 1000.0
    if u == "us":
        return value * 1_000_000.0
    return value


def _fmt_ns(ps: float) -> float:
    ns = ps / 1000.0
    rounded = round(ns, 3)
    if abs(rounded - round(rounded, 2)) < 1e-9:
        rounded = round(rounded, 2)
    if abs(rounded - int(rounded)) < 1e-9:
        return float(int(rounded))
    return float(rounded)


def _freq_mhz_from_period_ps(period_ps: float) -> float:
    if period_ps <= 0:
        return 0.0
    mhz = 1_000_000.0 / period_ps
    rounded = round(mhz, 2)
    if abs(rounded - int(rounded)) < 1e-9:
        return float(int(rounded))
    return float(rounded)


def _extract_brace_block(text: str, open_idx: int) -> Tuple[str, int]:
    """Return (inner, end_index_exclusive) for `{...}` starting at open_idx."""
    depth = 0
    i = open_idx
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
        i += 1
    return text[open_idx + 1 :], len(text)


def _parse_waveform_table_body(name: str, body: str, timing_name: str) -> Dict[str, Any]:
    period_match = _PERIOD_RE.search(body)
    period_ps = _to_ps(float(period_match.group(1)), period_match.group(2)) if period_match else 0.0

    edges_ps: List[float] = []
    for m in _EDGE_PS_RE.finditer(body):
        edges_ps.append(_to_ps(float(m.group(1)), m.group(2)))
    if period_ps and edges_ps and abs(edges_ps[0] - period_ps) < 1e-9:
        edges_ps = edges_ps[1:]

    strobe_ps: Optional[float] = None
    clock_rise_ps: Optional[float] = None
    clock_fall_ps: Optional[float] = None
    force_ps: Optional[float] = None

    for m in re.finditer(
        r"(_po_|LHXT|bidi_time_gen_\d+|input_time_gen_\d+)\s*\{([^}]*)\}",
        body,
        re.DOTALL,
    ):
        label = m.group(1)
        chunk = m.group(2)
        times = [_to_ps(float(a), b) for a, b in _EDGE_PS_RE.findall(chunk)]
        if not times:
            continue
        if "input_time" in label or label.startswith("input_"):
            force_ps = times[0]
        if "_po_" in label or "LHXT" in chunk:
            nonzero = [t for t in times if t > 0]
            if nonzero:
                strobe_ps = nonzero[0]
        if label.startswith("bidi_time_gen_1") or label.startswith("bidi_time_gen_2"):
            nonzero = [t for t in times if t > 0]
            if len(nonzero) >= 1:
                clock_rise_ps = nonzero[0]
            if len(nonzero) >= 2:
                clock_fall_ps = nonzero[1]

    unique = sorted({round(t, 3) for t in edges_ps if t > 0})
    if strobe_ps is None and unique:
        strobe_ps = unique[0]
    if clock_rise_ps is None and len(unique) >= 2:
        clock_rise_ps = unique[1]
    if clock_fall_ps is None and len(unique) >= 3:
        clock_fall_ps = unique[-1]
    if force_ps is None and unique:
        force_ps = min(unique)

    capture_spacing_ps: float = 0.0
    if clock_rise_ps is not None and strobe_ps is not None:
        capture_spacing_ps = abs(clock_rise_ps - strobe_ps)
    elif clock_rise_ps is not None and clock_fall_ps is not None:
        capture_spacing_ps = abs(clock_fall_ps - clock_rise_ps)
    elif period_ps:
        capture_spacing_ps = period_ps / 2.0

    return {
        "timingName": timing_name,
        "waveformTable": name,
        "periodPs": period_ps,
        "periodNs": _fmt_ns(period_ps) if period_ps else 0.0,
        "frequencyMhz": _freq_mhz_from_period_ps(period_ps) if period_ps else 0.0,
        "forcePs": force_ps,
        "strobePs": strobe_ps,
        "clockRisePs": clock_rise_ps,
        "clockFallPs": clock_fall_ps,
        "captureEdgeSpacingPs": capture_spacing_ps,
        "captureEdgeSpacingNs": _fmt_ns(capture_spacing_ps) if capture_spacing_ps else 0.0,
    }


@lru_cache(maxsize=2)
def load_stil_waveform_tables(stil_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return WaveformTable dicts sorted by frequency descending (fast → slow)."""
    path = stil_path or STIL_PATH
    if not path or not os.path.exists(path):
        return []

    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tables: List[Dict[str, Any]] = []
    for tm in re.finditer(r"Timing\s+(\w+)\s*\{", text):
        timing_name = tm.group(1)
        timing_body, _ = _extract_brace_block(text, tm.end() - 1)
        for wm in re.finditer(r"WaveformTable\s+(\w+)\s*\{", timing_body):
            body, _ = _extract_brace_block(timing_body, wm.end() - 1)
            tables.append(_parse_waveform_table_body(wm.group(1), body, timing_name))

    if not tables:
        for wm in re.finditer(r"WaveformTable\s+(\w+)\s*\{", text):
            body, _ = _extract_brace_block(text, wm.end() - 1)
            tables.append(_parse_waveform_table_body(wm.group(1), body, "Timing"))

    tables.sort(key=lambda t: (-float(t.get("frequencyMhz") or 0), str(t.get("waveformTable") or "")))
    return tables


def primary_waveform_table(stil_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    tables = load_stil_waveform_tables(stil_path)
    return tables[0] if tables else None


def half_rate_reference(table: Dict[str, Any]) -> Dict[str, Any]:
    """
    When only one WaveformTable / insertion exists, derive a slower half-rate
    reference from Period×2 for frequency-threshold comparison narrative.
    """
    period_ps = float(table.get("periodPs") or 0) * 2.0
    return {
        "timingName": table.get("timingName"),
        "waveformTable": f"{table.get('waveformTable')}_half_rate",
        "periodPs": period_ps,
        "periodNs": _fmt_ns(period_ps) if period_ps else 0.0,
        "frequencyMhz": _freq_mhz_from_period_ps(period_ps) if period_ps else 0.0,
        "forcePs": table.get("forcePs"),
        "strobePs": table.get("strobePs"),
        "clockRisePs": table.get("clockRisePs"),
        "clockFallPs": table.get("clockFallPs"),
        "captureEdgeSpacingPs": float(table.get("captureEdgeSpacingPs") or 0) * 2.0,
        "captureEdgeSpacingNs": _fmt_ns(float(table.get("captureEdgeSpacingPs") or 0) * 2.0),
        "derived": True,
        "source": "stil_period_x2",
    }


def clear_waveform_cache() -> None:
    load_stil_waveform_tables.cache_clear()
