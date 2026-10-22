"""
Compile training/inference dataset from scan debug data folder:
  - failure logs/
  - failing_bitmaps.json
  - diagnosis HTML report
"""

from __future__ import annotations

import json
import os
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.action_inference import infer_true_action
from src.data.chain_break_diagnosis import enrich_case_with_break_isolation
from src.data.paths import (
    COMPILED_DATASET_PATH,
    DATA_DIR,
    FAILURE_LOGS_DIR,
    FAILING_BITMAPS_PATH,
    diagnosis_html_path,
)


def clean_key(value: str) -> str:
    return str(value).strip().upper().replace("_", "").replace(" ", "").replace(".LOG", "")


class _TableCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: List[List[List[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_table: List[List[str]] = []
        self._current_row: List[str] = []
        self._cell_buf: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._current_row = []
        elif self._in_row and tag in ("td", "th"):
            self._in_cell = True
            self._cell_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._in_table:
            if self._current_table:
                self.tables.append(self._current_table)
            self._in_table = False
        elif tag == "tr" and self._in_row:
            if self._current_row:
                self._current_table.append(self._current_row)
            self._in_row = False
        elif tag in ("td", "th") and self._in_cell:
            self._current_row.append("".join(self._cell_buf).strip())
            self._in_cell = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_buf.append(data)


def _find_table_by_header(tables: List[List[List[str]]], header_fragment: str) -> Optional[List[List[str]]]:
    frag = header_fragment.lower()
    for table in tables:
        if not table:
            continue
        header = " ".join(table[0]).lower()
        if frag in header:
            return table
    return None


def _norm_header(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value or "")
    text = text.lower().strip()
    text = text.replace("±", "").replace("\ufffd", "")
    text = re.sub(r"\s+", " ", text)
    return text


def _header_index(header_row: List[str]) -> Dict[str, int]:
    return {_norm_header(h): i for i, h in enumerate(header_row)}


def _col_idx(index: Dict[str, int], *needles: str) -> Optional[int]:
    for name, idx in index.items():
        if all(needle in name for needle in needles):
            return idx
    return None


def _die_key(lot_id: str, die_label: str) -> str:
    die = str(die_label).replace(".log", "")
    return f"{clean_key(lot_id)}|{clean_key(die)}"


def _row_get(row: List[str], index: Dict[str, int], *needles: str, default: str = "") -> str:
    idx = _col_idx(index, *needles)
    if idx is None or idx >= len(row):
        return default
    return re.sub(r"<[^>]+>", "", row[idx]).strip() or default


def _parse_diagnosis_html(path: str) -> Tuple[
    Set[str],
    Dict[str, float],
    Dict[str, List[str]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    """Returns break_keys, min_slack_by_die, fault_models_by_die, chain_breaks_by_die, timing_by_die."""
    if not os.path.exists(path):
        return set(), {}, {}, {}, {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    parser = _TableCollector()
    parser.feed(html)
    tables = parser.tables

    break_keys: Set[str] = set()
    chain_breaks_by_die: Dict[str, Dict[str, Any]] = {}
    breaks_table = _find_table_by_header(tables, "candidate bit") or _find_table_by_header(tables, "lot id")
    if breaks_table:
        index = _header_index(breaks_table[0])
        for row in breaks_table[1:]:
            if len(row) < 3:
                continue
            lot_id = _row_get(row, index, "lot")
            die_file = _row_get(row, index, "die")
            key = _die_key(lot_id, die_file)
            break_keys.add(key)

            fail_count = int(re.sub(r"\D", "", _row_get(row, index, "fail", "count") or "0") or "0")
            conf_raw = _row_get(row, index, "confidence")
            exact_agree_raw = _row_get(row, index, "exact", "agree")
            candidate_cell = _row_get(row, index, "candidate", "cell")
            exact_bit = _row_get(row, index, "exact", "break", "bit")
            if exact_bit in ("—", "-", "LOCATION_UNCERTAIN", ""):
                exact_bit = ""

            parsed = {
                "chain_name": _row_get(row, index, "chain"),
                "location_status": _row_get(row, index, "location", "status"),
                "candidate_bit": _row_get(row, index, "candidate", "bit"),
                "candidate_cell": candidate_cell,
                "exact_break_bit": exact_bit or None,
                "exact_break_cell": _row_get(row, index, "exact", "break", "cell") or None,
                "break_confidence_pct": _pct_to_float(conf_raw),
                "exact_agree_pct": _pct_to_float(exact_agree_raw),
                "fail_count": fail_count,
                "scan_pins": _row_get(row, index, "scan"),
            }
            existing = chain_breaks_by_die.get(key)
            if not existing or parsed["fail_count"] >= existing.get("fail_count", 0):
                chain_breaks_by_die[key] = parsed

    slack_by_die: Dict[str, float] = {}
    faults_by_die: Dict[str, List[str]] = {}
    timing_by_die: Dict[str, Dict[str, Any]] = {}
    capture_table = _find_table_by_header(tables, "setup / hold slack")
    if capture_table:
        index = _header_index(capture_table[0])
        for row in capture_table[1:]:
            if len(row) < 7:
                continue
            lot_id = _row_get(row, index, "lot")
            die_file = _row_get(row, index, "die")
            key = _die_key(lot_id, die_file)

            slack_str = _row_get(row, index, "slack")
            classification = _row_get(row, index, "classification")
            try:
                parts = re.split(r"[/\s]+", slack_str.replace("—", "-"))
                for p in parts:
                    p = p.strip()
                    if p and re.match(r"^-?\d", p):
                        slack_by_die[key] = min(slack_by_die.get(key, 9999.0), float(p))
            except ValueError:
                pass

            if classification:
                faults_by_die.setdefault(key, [])
                if classification not in faults_by_die[key]:
                    faults_by_die[key].append(classification)

            violation = {
                "timing_chain": _row_get(row, index, "scan", "chain") or _row_get(row, index, "chain"),
                "timing_flop": _row_get(row, index, "failing", "cell"),
                "pattern_id": _row_get(row, index, "pattern"),
                "classification": classification,
                "slack": slack_str,
                "timing_detail": _row_get(row, index, "diagnostic"),
            }
            bucket = timing_by_die.setdefault(
                key,
                {
                    "timing_chain": violation["timing_chain"],
                    "timing_flop": violation["timing_flop"],
                    "timing_detail": violation["timing_detail"],
                    "timing_violations": [],
                },
            )
            bucket["timing_violations"].append(violation)
            if violation["timing_chain"]:
                bucket["timing_chain"] = violation["timing_chain"]
            if violation["timing_flop"]:
                bucket["timing_flop"] = violation["timing_flop"]
            if violation["timing_detail"]:
                bucket["timing_detail"] = violation["timing_detail"]

    return break_keys, slack_by_die, faults_by_die, chain_breaks_by_die, timing_by_die


def _pct_to_float(value: str) -> float:
    try:
        return float(str(value).replace("%", "").strip()) / 100.0
    except ValueError:
        return 0.0


def _parse_log_header(log_path: str) -> Dict[str, Any]:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(4000)
    fields: Dict[str, Any] = {}
    for key in (
        "LOT_ID",
        "DIE_LABEL",
        "DEFECT_TYPE",
        "TEST_MODE",
        "TEST_PROGRAM",
        "SHIFT_CYCLES",
        "CAPTURE_CYCLES",
        "SCAN_CHAINS",
        "TOTAL_PATTERNS",
        "DIE_ROW",
        "DIE_COL",
        "X1",
        "Y1",
        "X2",
        "Y2",
        "WAFER_X",
        "WAFER_Y",
    ):
        match = re.search(rf"{key}\s*:\s*(\S+)", head)
        if match:
            val = match.group(1)
            if key in (
                "SHIFT_CYCLES",
                "CAPTURE_CYCLES",
                "SCAN_CHAINS",
                "TOTAL_PATTERNS",
                "DIE_ROW",
                "DIE_COL",
                "X1",
                "Y1",
                "X2",
                "Y2",
                "WAFER_X",
                "WAFER_Y",
            ):
                try:
                    fields[key.lower()] = int(float(val))
                except ValueError:
                    fields[key.lower()] = val
            else:
                fields[key.lower()] = val
    return fields


def _parse_log_channel_failures(log_path: str) -> Dict[str, Any]:
    """Parse failing scan channels and pattern IDs from ATE pattern execution log."""
    failing_channels: List[str] = []
    failing_patterns: List[str] = []
    current_pattern: Optional[str] = None
    current_channel: Optional[str] = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            pattern_match = re.match(r"^(P\d+)\s*\|", line.strip())
            if pattern_match:
                current_pattern = pattern_match.group(1)
            channel_match = re.search(r"\bCH(\d+)\b", line)
            if channel_match and "EXPECTED_OUTPUT" in line:
                current_channel = f"channel{int(channel_match.group(1))}"
            if re.search(r"STATUS\s*:\s*F", line, flags=re.IGNORECASE):
                if current_channel and current_channel not in failing_channels:
                    failing_channels.append(current_channel)
                if current_pattern and current_pattern not in failing_patterns:
                    failing_patterns.append(current_pattern)

    return {
        "failing_channels": failing_channels,
        "failing_channel_count": len(failing_channels),
        "failing_patterns": failing_patterns[:20],
        "failing_pattern_count": len(failing_patterns),
    }


def _parse_log_mismatches(log_path: str) -> int:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return len(re.findall(r"STATUS\s*:\s*F", content, flags=re.IGNORECASE))


def _load_bitmap_index() -> Dict[str, Dict[str, Any]]:
    """Map LOT|DIE -> per-die wafer metadata from failing_bitmaps.json."""
    index: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(FAILING_BITMAPS_PATH):
        return index

    with open(FAILING_BITMAPS_PATH, "r", encoding="utf-8") as f:
        bitmap_doc = json.load(f)

    for lot in bitmap_doc.get("lots", []):
        lot_id = lot.get("lot_id", "")
        lot_fail_count = int(lot.get("fail_count", 0) or 0)
        for die in lot.get("dies", []):
            if die.get("status") != "fail":
                continue
            die_label = die.get("die_label", "")
            key = _die_key(lot_id, die_label)
            index[key] = {
                "lot_id": lot_id,
                "lot_folder": die.get("lot_folder") or lot.get("lot_folder"),
                "die_label": die_label,
                "defect_type": die.get("defect_type") or lot.get("defect_type"),
                "die_row": die.get("die_row"),
                "die_col": die.get("die_col"),
                "wafer_x": die.get("wafer_x"),
                "wafer_y": die.get("wafer_y"),
                "device_name": die.get("device_name"),
                "cell_count": 1,
                "lot_fail_count": lot_fail_count,
                "source_file": die.get("source_file"),
            }
    return index


def build_compiled_dataset(write: bool = True) -> List[Dict[str, Any]]:
    break_keys, slack_by_die, faults_by_die, chain_breaks, timing_by_die = _parse_diagnosis_html(
        diagnosis_html_path()
    )
    bitmap_index = _load_bitmap_index()
    from src.data.recommendation_engine import load_historical_index, match_historical_case

    hist_index = load_historical_index()
    cases: List[Dict[str, Any]] = []

    if not os.path.isdir(FAILURE_LOGS_DIR):
        raise FileNotFoundError(f"Failure logs directory not found: {FAILURE_LOGS_DIR}")

    for lot_folder in sorted(os.listdir(FAILURE_LOGS_DIR)):
        lot_path = os.path.join(FAILURE_LOGS_DIR, lot_folder)
        if not os.path.isdir(lot_path):
            continue
        for fname in sorted(os.listdir(lot_path)):
            if not fname.startswith("fail_die") or not fname.endswith(".log"):
                continue
            log_path = os.path.join(lot_path, fname)
            header = _parse_log_header(log_path)
            lot_id = header.get("lot_id") or lot_folder.split("_")[0]
            die_label = header.get("die_label") or fname.replace(".log", "")
            defect_type = header.get("defect_type") or "NORMAL"

            key = _die_key(lot_id, die_label)
            meta = bitmap_index.get(key, {})
            defect_type = meta.get("defect_type") or defect_type
            mismatch_count = _parse_log_mismatches(log_path)
            channel_info = _parse_log_channel_failures(log_path)
            has_break = key in break_keys
            min_slack = slack_by_die.get(key, 9999.0)
            fault_models = faults_by_die.get(key, [])
            cell_count = int(meta.get("cell_count", 1))
            break_info = chain_breaks.get(key, {})
            timing_info = timing_by_die.get(key, {})

            shifter_failure = has_break and defect_type.upper() in ("SCRATCH", "RANDOM")
            true_action = infer_true_action(
                defect_type=defect_type,
                has_break=has_break,
                min_slack=min_slack,
                mismatch_count=mismatch_count,
                cell_count=cell_count,
                fault_models=fault_models,
                shifter_failure=shifter_failure,
                chain_name=break_info.get("chain_name") or timing_info.get("timing_chain"),
                failing_channels=channel_info.get("failing_channels"),
            )

            case: Dict[str, Any] = {
                "lot_id": lot_id,
                "lot_folder": meta.get("lot_folder") or lot_folder,
                "die_label": die_label,
                "source_file": fname,
                "defect_type": defect_type,
                "mismatch_count": mismatch_count,
                "has_break": has_break,
                "min_slack": min_slack,
                "cell_count": cell_count,
                "fault_models": fault_models,
                "true_action": true_action,
                "wafer_x": header.get("wafer_x") or meta.get("wafer_x"),
                "wafer_y": header.get("wafer_y") or meta.get("wafer_y"),
                "die_row": meta.get("die_row") or header.get("die_row"),
                "die_col": meta.get("die_col") or header.get("die_col"),
                "shifter_failure": shifter_failure,
                "test_mode": header.get("test_mode"),
                "test_program": header.get("test_program") or meta.get("test_program"),
                "shift_cycles": header.get("shift_cycles"),
                "capture_cycles": header.get("capture_cycles"),
                "scan_chains": header.get("scan_chains"),
                "total_patterns": header.get("total_patterns"),
                "failing_channels": channel_info.get("failing_channels", []),
                "failing_channel_count": channel_info.get("failing_channel_count", 0),
                "failing_pattern_count": channel_info.get("failing_pattern_count", 0),
                "lot_fail_count": meta.get("lot_fail_count"),
                "bbox": {
                    "x1": header.get("x1"),
                    "y1": header.get("y1"),
                    "x2": header.get("x2"),
                    "y2": header.get("y2"),
                },
            }
            case.update({k: v for k, v in break_info.items() if v not in ("", None)})
            case.update({k: v for k, v in timing_info.items() if v not in ("", None) and k != "timing_violations"})
            if timing_info.get("timing_violations"):
                case["timing_violations"] = timing_info["timing_violations"]
            enrich_case_with_break_isolation(case)
            case["historical_match"] = match_historical_case(case, hist_index)

            cases.append(case)

    if write:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(COMPILED_DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2)

    return cases


if __name__ == "__main__":
    dataset = build_compiled_dataset(write=True)
    print(f"Compiled {len(dataset)} die cases -> {COMPILED_DATASET_PATH}")
