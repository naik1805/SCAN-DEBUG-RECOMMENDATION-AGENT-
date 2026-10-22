"""Tests for Power Debug Recommendations."""

from __future__ import annotations

import pytest

from src.data.dataset_builder import build_compiled_dataset
from src.data.power_debug_recs import (
    build_power_debug_recommendations,
    clear_power_debug_recs_caches,
    count_power_debug_recommendations,
    format_power_debug_recommendation,
    warm_power_debug_recs_cache,
)
from src.data.power_violations import clear_power_violation_caches


@pytest.fixture(scope="module")
def dataset():
    clear_power_violation_caches()
    clear_power_debug_recs_caches()
    return build_compiled_dataset(write=False)


def test_format_matches_product_example():
    text = format_power_debug_recommendation(
        pattern_id="4521",
        measured_mv=20.0,
        ir_threshold=15.0,
        prefer_ir=True,
        hist_count=3,
        hist_phrase=(
            "with similar IR-drop levels preceded test-only fails within ±5 patterns "
            "— recommend monitoring adjacent patterns"
        ),
    )
    assert "Check IR-drop during capture for Pattern #4521" in text
    assert "(20mV, 33% above threshold)" in text
    assert "3 historical cases with similar IR-drop levels" in text
    assert "within ±5 patterns" in text
    assert "recommend monitoring adjacent patterns" in text


def test_build_rows(dataset):
    clear_power_debug_recs_caches()
    rows = build_power_debug_recommendations(dataset)
    assert count_power_debug_recommendations(dataset) == len(rows)
    assert len(rows) >= 1
    for row in rows[:50]:
        assert "Check IR-drop during capture for Pattern #" in row["result"] or (
            "Check thermal during capture for Pattern #" in row["result"]
        )
        assert "above threshold" in row["result"]
        assert "historical case" in row["result"]
        assert "adjacent patterns" in row["result"]
        assert row["recommendedAction"] == "CHECK_IR_DROP_DURING_CAPTURE"
        assert row["historicalMatchCount"] >= 1
        assert row.get("pctAboveThreshold") is not None


def test_warm_cache(dataset):
    clear_power_debug_recs_caches()
    a = warm_power_debug_recs_cache(dataset)
    b = warm_power_debug_recs_cache(dataset)
    assert a == b
    assert a >= 1
