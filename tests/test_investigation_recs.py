"""Tests for Investigation Recommendations."""

from __future__ import annotations

import pytest

from src.data.dataset_builder import build_compiled_dataset
from src.data.investigation_recs import (
    build_investigation_recommendations,
    clear_investigation_recs_caches,
    count_investigation_recommendations,
    format_investigation_recommendation,
    get_investigation_recs_summary,
    warm_investigation_recs_cache,
)


@pytest.fixture(scope="module")
def dataset():
    clear_investigation_recs_caches()
    return build_compiled_dataset(write=False)


def test_format_matches_product_example():
    text = format_investigation_recommendation(
        net_id="N4521",
        neighbor_from="U890",
        neighbor_to="U912",
        fault_hypothesis="bridging fault",
        transition_fault_count=3,
        ir_drop_mv=8.0,
        ir_threshold_mv=15.0,
        power_induced_ruled_out=True,
        hist_count=3,
        hist_resolution="were resolved by SEM → FIB → PFA confirming bridging short",
    )
    assert text.startswith("Investigate Net N4521 (U890→U912) — suspected bridging fault.")
    assert "Confirmed real defect: transition faults present (count=3)" in text
    assert "normal IR-drop (8mV, below 15mV threshold)" in text
    assert "ruling out power-induced false fail" in text
    assert (
        "3 historical cases with matching diagnosis signature were resolved by "
        "SEM → FIB → PFA confirming bridging short"
    ) in text


def test_build_rows(dataset):
    clear_investigation_recs_caches()
    rows = build_investigation_recommendations(dataset)
    summary = get_investigation_recs_summary(dataset)
    assert count_investigation_recommendations(dataset) == int(summary["count"])
    assert len(rows) >= 1
    assert rows[0]["result"].startswith("Investigate Net ")
    assert "suspected" in rows[0]["result"]
    assert "historical case" in rows[0]["result"]
    assert rows[0]["recommendedAction"] == "INVESTIGATE_PHYSICAL_DEFECT"
    assert rows[0]["faultHypothesis"]
    assert rows[0].get("irThresholdMv") == 15.0
    assert "transitionFaultCount" in rows[0]


def test_warm_cache(dataset):
    clear_investigation_recs_caches()
    a = warm_investigation_recs_cache(dataset)
    b = warm_investigation_recs_cache(dataset)
    assert a == b
    assert a >= 1
