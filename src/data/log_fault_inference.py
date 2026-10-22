"""Infer stuck-at fault type from ATE failure log expected vs observed waveforms."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.data.paths import FAILURE_LOGS_DIR
from src.data.stil_parser import chain_display_number

_WAVE_EXPAND_RE = re.compile(r"X@\{(\d+)\}")
_HL_TO_BIT = {"H": "1", "L": "0", "1": "1", "0": "0"}


def expand_ate_waveform(wave: str) -> List[str]:
    """Expand X@{n} and return per-cycle symbols (H/L/X/...)."""
    out: List[str] = []
    i = 0
    text = wave.strip()
    while i < len(text):
        m = _WAVE_EXPAND_RE.match(text, i)
        if m:
            out.extend(["X"] * int(m.group(1)))
            i += m.end() - m.start()
        else:
            out.append(text[i])
            i += 1
    return out


def _log_path_for_case(case: Dict[str, Any]) -> Optional[str]:
    lot_folder = case.get("lot_folder")
    source = case.get("source_file")
    if not lot_folder or not source:
        return None
    path = os.path.join(FAILURE_LOGS_DIR, lot_folder, source)
    return path if os.path.exists(path) else None


def _compare_at_bit(expected_sym: str, actual_sym: str) -> Optional[str]:
    exp = _HL_TO_BIT.get(expected_sym.upper())
    act = _HL_TO_BIT.get(actual_sym.upper())
    if exp is None or act is None or exp == act:
        return None
    if exp == "1" and act == "0":
        return "stuck-at-0"
    if exp == "0" and act == "1":
        return "stuck-at-1"
    return None


def parse_failing_waveform_pairs(log_path: str, channel_name: str) -> List[Tuple[List[str], List[str]]]:
    """Parse failing (expected, actual) waveform pairs for one scan channel."""
    ch_num = chain_display_number(channel_name)
    if ch_num is None:
        return []

    ch_re = re.compile(rf"\bCH{ch_num}\b")
    pairs: List[Tuple[List[str], List[str]]] = []
    pending_exp: Optional[str] = None
    pending_act: Optional[str] = None
    pending_for_channel = False

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if ch_re.search(s) and "EXPECTED_OUTPUT:" in s:
                pending_exp = s.split("EXPECTED_OUTPUT:", 1)[1].strip()
                pending_act = None
                pending_for_channel = True
                continue

            if not pending_for_channel:
                continue

            if "ACTUAL_OUTPUT:" in s:
                pending_act = s.split("ACTUAL_OUTPUT:", 1)[1].strip()
                continue

            if re.search(r"STATUS\s*:\s*F", s, re.I):
                if pending_exp and pending_act:
                    pairs.append(
                        (expand_ate_waveform(pending_exp), expand_ate_waveform(pending_act))
                    )
                pending_exp = None
                pending_act = None
                pending_for_channel = False
                continue

            if ch_re.search(s) or (s.startswith("P") and "|" in s and "CH" in s):
                pending_exp = None
                pending_act = None
                pending_for_channel = False

    return pairs


def infer_fault_type_from_log(
    case: Dict[str, Any],
    candidate_bit: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Infer stuck-at-0 / stuck-at-1 from log expected vs observed at diagnosis bit.
    Returns (fault_type, meta).
    """
    meta: Dict[str, Any] = {"source": "default_stil", "votes": {}}
    log_path = _log_path_for_case(case)
    chain = case.get("chain_name") or ""

    bit = candidate_bit
    if bit is None:
        try:
            bit = int(str(case.get("candidate_bit") or "").strip())
        except (TypeError, ValueError):
            bit = None

    if not log_path or not chain:
        return "stuck-at-0", meta

    pairs = parse_failing_waveform_pairs(log_path, chain)
    if not pairs:
        return "stuck-at-0", meta

    votes: Dict[str, int] = {"stuck-at-0": 0, "stuck-at-1": 0}
    comparisons = 0

    for exp, act in pairs:
        n = min(len(exp), len(act))
        if n == 0:
            continue

        indices: List[int] = []
        if bit is not None and 0 <= bit < n:
            indices = [bit]
        else:
            for i in range(n):
                if _compare_at_bit(exp[i], act[i]):
                    indices = [i]
                    break

        for i in indices:
            fault = _compare_at_bit(exp[i], act[i])
            if fault:
                votes[fault] += 1
                comparisons += 1

    meta["votes"] = votes
    meta["comparisons"] = comparisons
    meta["patterns_compared"] = len(pairs)

    if comparisons == 0:
        return "stuck-at-0", meta

    fault = max(votes, key=lambda k: votes[k])
    meta["source"] = "failure_log"
    return fault, meta
