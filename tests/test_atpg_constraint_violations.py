"""Tests for STIL held-pin × failing-pattern over-constraint detection."""

from __future__ import annotations

import pytest

from src.data.atpg_constraint_violations import (
    build_constraint_violation_results,
    clear_constraint_violation_caches,
    compute_constraint_violation,
    constraint_violation_category_counts,
    count_constraint_violations,
    format_over_constraint_result,
    held_pin_constraint_types,
    warm_constraint_violation_cache,
)
from src.data.dataset_builder import build_compiled_dataset
from src.data.stil_constraint_parser import format_held_pins


@pytest.fixture(scope="module")
def dataset():
    clear_constraint_violation_caches()
    return build_compiled_dataset(write=False)


@pytest.fixture(scope="module")
def fail_cases(dataset):
    return [c for c in dataset if str(c.get("source_file", "")).startswith("fail")]


def test_format_held_pins_includes_clock_reset_test():
    held = format_held_pins("01100000")
    assert "XI=0" in held
    assert "TEST=1" in held
    assert "RESET_N=1" in held


def test_held_pin_constraint_types():
    typed = held_pin_constraint_types("XI=0,TEST=1,RESET_N=1")
    cats = {t["constraintCategory"] for t in typed}
    assert cats == {"clock", "scan_enable", "reset"}


def test_format_over_constraint_result_categorized():
    text = format_over_constraint_result(
        "RESET_N=1", "SPI0_IO2_WP", 40, category_label="Reset"
    )
    assert "Reset constraint violation" in text
    assert "RESET_N=1" in text
    assert "40 patterns" in text


def test_compute_constraint_violation_on_fail_die(fail_cases):
    case = next((c for c in fail_cases if c.get("lot_id") == "LOT_7"), fail_cases[0])
    row = compute_constraint_violation(case)
    assert row is not None
    assert row["suspectedOverConstraint"] is True
    assert row["failingPatternCount"] >= 3
    assert row["fanoutSignal"]
    assert len(row["constraintTypes"]) >= 2


def test_count_is_typed_not_dies(fail_cases):
    clear_constraint_violation_caches()
    rows = build_constraint_violation_results(fail_cases)
    count = count_constraint_violations(fail_cases)
    assert count == len(rows)
    assert count >= 15
    keys = {
        (r["constraintCategory"], r["heldPins"], r["fanoutSignal"]) for r in rows
    }
    assert len(keys) == count
    cats = constraint_violation_category_counts(fail_cases)
    assert "reset" in cats
    assert "scan_enable" in cats
    assert "clock" in cats


def test_warm_cache_is_fast_second_call(fail_cases):
    clear_constraint_violation_caches()
    first = warm_constraint_violation_cache(fail_cases)
    second = warm_constraint_violation_cache(fail_cases)
    assert first == second
    assert first >= 15
