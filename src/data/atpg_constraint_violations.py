"""Detect suspected ATPG over-constraints via STIL held pins × failing-pattern fan-out.

Results are persisted to disk so dashboard / KPI endpoints stay fast.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.paths import (
    COMPILED_DATASET_PATH,
    CONSTRAINT_VIOLATIONS_PATH,
    FAILURE_LOGS_DIR,
    STIL_PATH,
)
from src.data.stil_constraint_parser import format_held_pins
from src.data.stil_parser import load_stil_scan_chains

_LOG_PATTERN_CH_RE = re.compile(r"^(P\d+)\s*\|\s*CH(\d+)\b")
_FAIL_STATUS_RE = re.compile(r"STATUS\s*:\s*F", re.IGNORECASE)
_PI_VALUE_RE = re.compile(r"_pi_=([01X]+)")
_PROCEDURE_RE = re.compile(r"procedure\s*=\s*([A-Za-z0-9_]+)")

_VIOLATION_ROWS_CACHE: Optional[Tuple[str, List[Dict[str, Any]]]] = None

# STIL held-pin → ATPG constraint category (engineer-facing taxonomy).
_PIN_CONSTRAINT_CATEGORY: Dict[str, str] = {
    "XI": "clock",
    "TEST": "scan_enable",
    "RESET_N": "reset",
    "RESETN": "reset",
    "BSEN": "scan_enable",
}

_CATEGORY_LABELS: Dict[str, str] = {
    "clock": "Clock",
    "scan_enable": "Scan Enable",
    "reset": "Reset",
}

_CATEGORY_ORDER = ("reset", "scan_enable", "clock")


def _inputs_fingerprint() -> str:
    parts: List[str] = ["v4-full-pattern-ids"]
    for path in (COMPILED_DATASET_PATH, STIL_PATH):
        if path and os.path.exists(path):
            parts.append(f"{path}:{os.path.getmtime(path)}")
    return "|".join(parts)


@lru_cache(maxsize=1)
def _stil_held_pins_and_procedure() -> Tuple[str, str]:
    """Read first STIL procedure/_pi_ vector only — avoids full pattern catalog parse."""
    held = format_held_pins("01100000")
    procedure = "scan_shift_capture"
    if not STIL_PATH or not os.path.exists(STIL_PATH):
        return _compact_held_pins(held), procedure

    found_pi = False
    found_proc = False
    with open(STIL_PATH, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not found_proc:
                proc_match = _PROCEDURE_RE.search(line)
                if proc_match:
                    procedure = proc_match.group(1)
                    found_proc = True
            if not found_pi:
                pi_match = _PI_VALUE_RE.search(line)
                if pi_match:
                    held = format_held_pins(pi_match.group(1))
                    found_pi = True
            if found_pi and found_proc:
                break
    return _compact_held_pins(held), procedure


def _compact_held_pins(held_pins: str) -> str:
    """Keep constraint-relevant held pins: clock (XI), scan-enable (TEST), reset (RESET_N)."""
    tokens = [
        t
        for t in held_pins.split(",")
        if t.startswith(("XI=", "RESET_N=", "TEST=", "BSEN="))
    ]
    return ",".join(tokens) if tokens else held_pins


def categorize_held_pin(pin_token: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse 'RESET_N=1' → (category_id, pin_name, value) e.g. ('reset', 'RESET_N', '1').
    """
    if "=" not in pin_token:
        return None
    name, value = pin_token.split("=", 1)
    category = _PIN_CONSTRAINT_CATEGORY.get(name)
    if not category:
        upper = name.upper()
        if "RESET" in upper or upper.endswith("RST") or upper.endswith("_RST"):
            category = "reset"
        elif "CLK" in upper or upper in ("XI", "TCK"):
            category = "clock"
        elif upper in ("TEST", "SE", "BSEN") or "SCAN_EN" in upper:
            category = "scan_enable"
        else:
            return None
    return category, name, value


def held_pin_constraint_types(held_pins: str) -> List[Dict[str, str]]:
    """Expand held-pin string into typed constraint entries."""
    out: List[Dict[str, str]] = []
    for token in held_pins.split(","):
        token = token.strip()
        if not token:
            continue
        parsed = categorize_held_pin(token)
        if not parsed:
            continue
        category, pin, value = parsed
        out.append(
            {
                "constraintCategory": category,
                "constraintCategoryLabel": _CATEGORY_LABELS.get(category, category),
                "constraintPin": pin,
                "constraintValue": value,
                "heldPinToken": f"{pin}={value}",
            }
        )
    # Stable order: reset, scan_enable, clock
    out.sort(key=lambda e: _CATEGORY_ORDER.index(e["constraintCategory"]) if e["constraintCategory"] in _CATEGORY_ORDER else 99)
    return out


def format_over_constraint_result(
    held_pin_token: str,
    fanout: str,
    pattern_count: int,
    *,
    category_label: str,
    procedure: str = "scan_shift_capture",
) -> str:
    return (
        f"{category_label} constraint violation ({held_pin_token}): held per STIL procedure, "
        f"correlates with fails on {pattern_count} patterns exercising {fanout} fan-out "
        f"— suspected over-constraint"
    )


def _log_path_for_case(case: Dict[str, Any]) -> Optional[str]:
    lot_folder = case.get("lot_folder")
    source = case.get("source_file")
    if not lot_folder or not source:
        return None
    path = os.path.join(FAILURE_LOGS_DIR, lot_folder, source)
    return path if os.path.exists(path) else None


@lru_cache(maxsize=256)
def failing_pattern_channel_pairs(log_path: str) -> Tuple[Tuple[str, str], ...]:
    """
    Return (pattern_label, fanout_scan_in) for each failing pattern/channel in log.
    Cached per log path for repeated dashboard / workspace calls.
    """
    chains = load_stil_scan_chains()
    pairs: List[Tuple[str, str]] = []
    current_pat: Optional[str] = None
    current_ch: Optional[str] = None

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = _LOG_PATTERN_CH_RE.match(line.strip())
            if match:
                current_pat = match.group(1)
                current_ch = match.group(2)
            if _FAIL_STATUS_RE.search(line) and current_pat and current_ch:
                ch_key = f"channel{int(current_ch)}"
                fanout = (chains.get(ch_key) or {}).get("scan_in") or ch_key
                pairs.append((current_pat, fanout))

    return tuple(pairs)


def failing_pattern_labels(log_path: str) -> Set[str]:
    return {pat for pat, _ in failing_pattern_channel_pairs(log_path)}


def _dominant_fanout(
    pairs: Tuple[Tuple[str, str], ...],
) -> Optional[Tuple[str, int, Set[str]]]:
    fanout_hits: Counter[str] = Counter()
    fanout_patterns: Dict[str, Set[str]] = defaultdict(set)
    for label, fanout in pairs:
        fanout_hits[fanout] += 1
        fanout_patterns[fanout].add(label)
    if not fanout_hits:
        return None
    fanout, count = fanout_hits.most_common(1)[0]
    if count < 3:
        return None
    return fanout, count, fanout_patterns[fanout]


def compute_constraint_violation(case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Analyze one die for a dominant fan-out cluster under STIL held pins."""
    log_path = _log_path_for_case(case)
    if not log_path:
        return None

    pairs = failing_pattern_channel_pairs(log_path)
    failing = {pat for pat, _ in pairs}
    if len(failing) < 3:
        return None

    dominant = _dominant_fanout(pairs)
    if not dominant:
        return None

    fanout, pattern_count, pattern_labels = dominant
    held_pins, procedure = _stil_held_pins_and_procedure()
    typed = held_pin_constraint_types(held_pins)
    if not typed:
        return None

    return {
        "heldPins": held_pins,
        "fanoutSignal": fanout,
        "failingPatternCount": pattern_count,
        "totalFailingPatterns": len(failing),
        "failingChannelObservations": len(pairs),
        "procedure": procedure,
        "patternLabels": sorted(pattern_labels),
        "lotId": case.get("lot_id"),
        "dieLabel": case.get("die_label"),
        "constraintTypes": typed,
        "suspectedOverConstraint": True,
    }


def _expand_typed_violations(base: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One violation row per constraint category (reset / scan_enable / clock)."""
    rows: List[Dict[str, Any]] = []
    for typed in base.get("constraintTypes") or []:
        category = typed["constraintCategory"]
        label = typed["constraintCategoryLabel"]
        pin_token = typed["heldPinToken"]
        result = format_over_constraint_result(
            pin_token,
            base["fanoutSignal"],
            base["failingPatternCount"],
            category_label=label,
            procedure=base.get("procedure") or "scan_shift_capture",
        )
        rows.append(
            {
                "result": result,
                "heldPins": pin_token,
                "allHeldPins": base.get("heldPins"),
                "fanoutSignal": base["fanoutSignal"],
                "failingPatternCount": base["failingPatternCount"],
                "totalFailingPatterns": base["totalFailingPatterns"],
                "failingChannelObservations": base["failingChannelObservations"],
                "procedure": base.get("procedure"),
                "patternLabels": base.get("patternLabels"),
                "lotId": base.get("lotId"),
                "dieLabel": base.get("dieLabel"),
                "constraintCategory": category,
                "constraintCategoryLabel": label,
                "constraintPin": typed["constraintPin"],
                "constraintValue": typed["constraintValue"],
                "suspectedOverConstraint": True,
            }
        )
    return rows


def _load_persisted(fingerprint: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(CONSTRAINT_VIOLATIONS_PATH):
        return None
    try:
        with open(CONSTRAINT_VIOLATIONS_PATH, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("fingerprint") != fingerprint:
            return None
        rows = doc.get("rows")
        return rows if isinstance(rows, list) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _persist(fingerprint: str, rows: List[Dict[str, Any]]) -> None:
    try:
        os.makedirs(os.path.dirname(CONSTRAINT_VIOLATIONS_PATH), exist_ok=True)
        by_cat = Counter(r.get("constraintCategory") or "other" for r in rows)
        with open(CONSTRAINT_VIOLATIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "artifact": "constraint_violations",
                    "fingerprint": fingerprint,
                    "count": len(rows),
                    "byCategory": dict(by_cat),
                    "rows": rows,
                },
                f,
                indent=2,
            )
    except OSError:
        pass


def build_constraint_violation_results(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    One row per distinct typed over-constraint:
    (constraint category × held pin × fan-out), not one row per die.
    Categories: reset, scan_enable, clock.
    """
    global _VIOLATION_ROWS_CACHE
    fingerprint = _inputs_fingerprint()
    if _VIOLATION_ROWS_CACHE is not None and _VIOLATION_ROWS_CACHE[0] == fingerprint:
        return _VIOLATION_ROWS_CACHE[1]

    persisted = _load_persisted(fingerprint)
    if persisted is not None:
        _VIOLATION_ROWS_CACHE = (fingerprint, persisted)
        return persisted

    by_key: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        if not str(case.get("source_file", "")).startswith("fail"):
            continue
        base = compute_constraint_violation(case)
        if not base:
            continue
        for row in _expand_typed_violations(base):
            key = f"{row['constraintCategory']}|{row['heldPins']}|{row['fanoutSignal']}"
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = {
                    **row,
                    "affectedDies": 1,
                    "dieLabels": [f"{row.get('lotId')}/{row.get('dieLabel')}"],
                    "lotIds": [row.get("lotId")],
                }
                continue

            existing["affectedDies"] = int(existing.get("affectedDies") or 0) + 1
            existing["dieLabels"] = list(existing.get("dieLabels") or []) + [
                f"{row.get('lotId')}/{row.get('dieLabel')}"
            ]
            lots = list(existing.get("lotIds") or [])
            if row.get("lotId") not in lots:
                lots.append(row.get("lotId"))
            existing["lotIds"] = lots
            if row.get("failingPatternCount", 0) > existing.get("failingPatternCount", 0):
                existing["failingPatternCount"] = row["failingPatternCount"]
                existing["totalFailingPatterns"] = row["totalFailingPatterns"]
                existing["failingChannelObservations"] = row["failingChannelObservations"]
                existing["patternLabels"] = row["patternLabels"]
                existing["lotId"] = row.get("lotId")
                existing["dieLabel"] = row.get("dieLabel")
                existing["result"] = format_over_constraint_result(
                    row["heldPins"],
                    row["fanoutSignal"],
                    row["failingPatternCount"],
                    category_label=row["constraintCategoryLabel"],
                    procedure=row.get("procedure") or "scan_shift_capture",
                )

    rows = list(by_key.values())
    rows.sort(
        key=lambda r: (
            _CATEGORY_ORDER.index(r["constraintCategory"])
            if r.get("constraintCategory") in _CATEGORY_ORDER
            else 99,
            -r.get("failingPatternCount", 0),
            -r.get("affectedDies", 0),
            str(r.get("fanoutSignal") or ""),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["dieLabels"] = (row.get("dieLabels") or [])[:12]

    _persist(fingerprint, rows)
    _VIOLATION_ROWS_CACHE = (fingerprint, rows)
    return rows


def count_constraint_violations(cases: List[Dict[str, Any]]) -> int:
    """Count distinct typed over-constraints (category × pin × fan-out)."""
    return len(build_constraint_violation_results(cases))


def constraint_violation_category_counts(
    cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return {}
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if not isinstance(cases, list):
            return {}
    rows = build_constraint_violation_results(cases)
    return dict(Counter(r.get("constraintCategory") or "other" for r in rows))


def warm_constraint_violation_cache(cases: Optional[List[Dict[str, Any]]] = None) -> int:
    """Precompute + persist so dashboard requests stay instant."""
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return 0
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if not isinstance(cases, list):
            return 0
    return count_constraint_violations(cases)


def clear_constraint_violation_caches() -> None:
    global _VIOLATION_ROWS_CACHE
    _VIOLATION_ROWS_CACHE = None
    failing_pattern_channel_pairs.cache_clear()
    _stil_held_pins_and_procedure.cache_clear()
