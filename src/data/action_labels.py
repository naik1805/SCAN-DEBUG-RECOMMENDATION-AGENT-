"""Human-readable debug action labels for API and UI."""

from __future__ import annotations

from typing import Dict

# Internal training / env labels (compiled dataset true_action)
INTERNAL_ACTION_LABELS: Dict[str, str] = {
    "INSPECT_SCAN_CHAIN": "Inspect Scan Chain",
    "REVIEW_CAPTURE_CLOCK_TIMING": "Review Capture Clock Timing",
    "CHECK_IR_DROP_DURING_CAPTURE": "Check IR-Drop During Capture",
    "REVIEW_ATPG_CONSTRAINTS": "Review ATPG Constraints",
    "INVESTIGATE_PHYSICAL_DEFECT": "Investigate Physical Defect",
}

# UI / DQN category codes used in dashboard payloads
UI_ACTION_LABELS: Dict[str, str] = {
    "SCAN_CHAIN_DEBUG": "Inspect Scan Chain",
    "TIMING_DEBUG": "Review Capture Clock Timing",
    "POWER_RELATED_DEBUG": "Check IR-Drop During Capture",
    "ATPG_CONSTRAINT_REVIEW": "Review ATPG Constraints",
    "PHYSICAL_DEFECT_INVESTIGATION": "Investigate Physical Defect",
}

UI_ACTION_MAP = {
    "INSPECT_SCAN_CHAIN": "SCAN_CHAIN_DEBUG",
    "REVIEW_CAPTURE_CLOCK_TIMING": "TIMING_DEBUG",
    "CHECK_IR_DROP_DURING_CAPTURE": "POWER_RELATED_DEBUG",
    "REVIEW_ATPG_CONSTRAINTS": "ATPG_CONSTRAINT_REVIEW",
    "INVESTIGATE_PHYSICAL_DEFECT": "PHYSICAL_DEFECT_INVESTIGATION",
}

CATEGORY_COLORS = {
    "SCAN_CHAIN_DEBUG": "#7C3AED",
    "TIMING_DEBUG": "#F59E0B",
    "POWER_RELATED_DEBUG": "#EF4444",
    "ATPG_CONSTRAINT_REVIEW": "#38BDF8",
    "PHYSICAL_DEFECT_INVESTIGATION": "#10B981",
}


def action_label_from_internal(action: str) -> str:
    return INTERNAL_ACTION_LABELS.get(action, action.replace("_", " ").title())


def action_label_from_category(category: str) -> str:
    return UI_ACTION_LABELS.get(category, category.replace("_", " ").title())


def category_from_internal(action: str) -> str:
    return UI_ACTION_MAP.get(action, "ATPG_CONSTRAINT_REVIEW")
