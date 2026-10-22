"""Tests for Timing Debug Recommendations."""

from __future__ import annotations

import pytest

from src.data.dataset_builder import build_compiled_dataset
from src.data.timing_debug_recs import (
    build_timing_debug_recommendations,
    clear_timing_debug_recs_caches,
    count_timing_debug_recommendations,
    diagnosis_flags_transition_path_delay,
    format_timing_debug_recommendation,
    warm_timing_debug_recs_cache,
)
from src.data.timing_violations import clear_timing_violation_caches


@pytest.fixture(scope="module")
def dataset():
    clear_timing_violation_caches()
    clear_timing_debug_recs_caches()
    return build_compiled_dataset(write=False)


def test_format_matches_product_example():
    text = format_timing_debug_recommendation(
        pattern_id="2210",
        chain="channel21",
        clock_domain="ETH_RXCLK",
        hist_count=5,
        hist_phrase="with frequency",
    )
    assert "Review capture clock timing for pattern #2210's capture window" in text
    assert "channel21" in text
    assert "ETH_RXCLK" in text
    assert "5 historical cases with frequency" in text


def test_diagnosis_transition_path_delay_flags():
    assert diagnosis_flags_transition_path_delay("CAPTURE_TIMING_SETUP")
    assert diagnosis_flags_transition_path_delay("CAPTURE_TIMING_SETUP_ANOMALY")
    assert diagnosis_flags_transition_path_delay("PATH_DELAY_FAULT")
    assert not diagnosis_flags_transition_path_delay("CAPTURE_TIMING_HOLD")


def test_build_rows(dataset):
    clear_timing_debug_recs_caches()
    rows = build_timing_debug_recommendations(dataset)
    assert count_timing_debug_recommendations(dataset) == len(rows)
    assert len(rows) >= 1
    for row in rows:
        assert "Review capture clock timing for pattern #" in row["result"]
        assert "capture window" in row["result"]
        assert "historical case" in row["result"]
        assert row["recommendedAction"] == "REVIEW_CAPTURE_CLOCK_TIMING"
        assert row["clockDomain"]
        assert row["timingChain"]
        assert row["historicalMatchCount"] >= 1


def test_warm_cache(dataset):
    clear_timing_debug_recs_caches()
    a = warm_timing_debug_recs_cache(dataset)
    b = warm_timing_debug_recs_cache(dataset)
    assert a == b
    assert a >= 1
