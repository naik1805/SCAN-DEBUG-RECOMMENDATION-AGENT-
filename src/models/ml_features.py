"""Feature extraction from compiled die cases for supervised ML recommenders."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np

from src.env.state_builder import DEFECT_MAP

FEATURE_NAMES: Tuple[str, ...] = (
    "mismatch_norm",
    "has_break",
    "shifter_failure",
    "min_slack_norm",
    "cell_count_norm",
    "fail_count_norm",
    "failing_channel_norm",
    "failing_pattern_norm",
    "wafer_x_norm",
    "wafer_y_norm",
    "defect_encoding",
    "timing_fault_flag",
    "stuck_fault_flag",
    "transition_fault_flag",
    "ir_fault_flag",
    "chain_present",
    "hist_confidence",
)


def _norm_slack(slack: float) -> float:
    if slack >= 9999:
        return 1.0
    clipped = max(-100.0, min(100.0, float(slack)))
    return (clipped + 100.0) / 200.0


def _fault_flags(fault_models: List[str]) -> Tuple[float, float, float, float]:
    faults = [str(f).lower() for f in fault_models or []]
    timing = any(
        t in f
        for f in faults
        for t in ("transition", "hold", "setup", "timing", "delay", "slack")
    )
    stuck = any(s in f for f in faults for s in ("stuck", "stuck-at", "sa0", "sa1"))
    transition = any("transition" in f for f in faults)
    ir = any("ir" in f or "power" in f or "thermal" in f for f in faults)
    return (
        1.0 if timing else 0.0,
        1.0 if stuck else 0.0,
        1.0 if transition else 0.0,
        1.0 if ir else 0.0,
    )


def case_feature_vector(case: Dict[str, Any]) -> np.ndarray:
    """Numeric feature vector for one compiled die case."""
    mismatch = float(case.get("mismatch_count") or case.get("fail_count") or 0)
    fail_count = float(case.get("fail_count") or mismatch)
    timing, stuck, transition, ir = _fault_flags(case.get("fault_models") or [])
    defect = str(case.get("defect_type") or "NORMAL").upper()
    hist = case.get("historical_match") or {}

    wx = float(case.get("wafer_x") or 0)
    wy = float(case.get("wafer_y") or 0)

    return np.array(
        [
            min(mismatch / 100.0, 1.0),
            1.0 if case.get("has_break") else 0.0,
            1.0 if case.get("shifter_failure") else 0.0,
            _norm_slack(float(case.get("min_slack", 9999))),
            min(float(case.get("cell_count") or 0) / 100.0, 1.0),
            min(fail_count / 100.0, 1.0),
            min(float(case.get("failing_channel_count") or 0) / 10.0, 1.0),
            min(float(case.get("failing_pattern_count") or 0) / 100.0, 1.0),
            min(wx / 200.0, 1.0),
            min(wy / 200.0, 1.0),
            DEFECT_MAP.get(defect, 0.9),
            timing,
            stuck,
            transition,
            ir,
            1.0 if case.get("chain_name") else 0.0,
            float(hist.get("confidence") or 0.0),
        ],
        dtype=np.float32,
    )


def cases_feature_matrix(cases: List[Dict[str, Any]]) -> np.ndarray:
    if not cases:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([case_feature_vector(c) for c in cases])
