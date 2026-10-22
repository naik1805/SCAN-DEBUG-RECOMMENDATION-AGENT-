"""Infer expert debug action labels from die features (shared by env + dataset builder)."""

from __future__ import annotations

from typing import List, Optional

ACTION_LABELS = {
    0: "INSPECT_SCAN_CHAIN",
    1: "REVIEW_CAPTURE_CLOCK_TIMING",
    2: "CHECK_IR_DROP_DURING_CAPTURE",
    3: "REVIEW_ATPG_CONSTRAINTS",
    4: "INVESTIGATE_PHYSICAL_DEFECT",
}

DEFECT_ENC = {
    "CENTER": 0.1,
    "NEAR_FULL": 0.2,
    "LOCAL": 0.3,
    "DONUT": 0.4,
    "EDGE_RING": 0.5,
    "EDGE_LOC": 0.6,
    "SCRATCH": 0.7,
    "RANDOM": 0.8,
    "NORMAL": 0.9,
}


def infer_true_action(
    defect_type: str,
    has_break: bool,
    min_slack: float,
    mismatch_count: int,
    cell_count: int,
    fault_models: List[str] | None = None,
    shifter_failure: bool = False,
    chain_name: Optional[str] = None,
    failing_channels: List[str] | None = None,
) -> str:
    """Evidence-based action from parsed diagnosis, logs, and defect signature."""
    faults = [f.lower() for f in (fault_models or [])]
    has_timing = any(
        t in f for f in faults for t in ["transition", "hold", "setup", "timing", "delay", "capture_timing"]
    )
    has_stuck = any(s in f for f in faults for s in ["stuck", "stuck-at", "sa0", "sa1"])

    defect_key = (defect_type or "NORMAL").upper().replace("-", "_")
    defect_enc = DEFECT_ENC.get(defect_key, 0.9)

    chain = (chain_name or "").lower()
    failing = [c.lower() for c in (failing_channels or [])]
    chain_in_log_failures = chain and chain in failing

    # 1. Timing debug — capture slack / timing fault models from diagnosis registry
    if min_slack < 9999.0 or has_timing:
        return ACTION_LABELS[1]

    # 2. Scan chain debug — scratch/random defect lots and explicit shift failures
    if shifter_failure:
        return ACTION_LABELS[0]
    if defect_key in ("SCRATCH", "RANDOM") and has_break:
        return ACTION_LABELS[0]
    if has_stuck and has_break and defect_key in ("SCRATCH", "RANDOM"):
        return ACTION_LABELS[0]

    # 3. Power-related debug — center / near-full wafer defect signatures
    if defect_key in ("CENTER", "NEAR_FULL"):
        return ACTION_LABELS[2]

    # 4. Physical defect investigation — localized / edge / donut patterns
    if defect_key in ("LOCAL", "DONUT", "EDGE_RING", "EDGE_LOC", "EDGE-RING", "EDGE-LOC"):
        return ACTION_LABELS[4]

    # 5. ATPG constraint review — NORMAL lots (e.g. LOT_7) and unclassified mismatches
    if defect_key == "NORMAL":
        return ACTION_LABELS[3]

    return ACTION_LABELS[3]
