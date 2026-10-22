"""Tests for Timing Violations (at-speed / WaveformTable margin)."""

from __future__ import annotations

import pytest

from src.data.dataset_builder import build_compiled_dataset
from src.data.stil_waveform_parser import (
    clear_waveform_cache,
    half_rate_reference,
    load_stil_waveform_tables,
)
from src.data.timing_violations import (
    build_timing_violation_results,
    clear_timing_violation_caches,
    count_timing_violations,
    format_timing_violation_result,
    warm_timing_violation_cache,
)


@pytest.fixture(scope="module")
def dataset():
    clear_waveform_cache()
    clear_timing_violation_caches()
    return build_compiled_dataset(write=False)


def test_stil_waveform_table_period_and_spacing():
    clear_waveform_cache()
    tables = load_stil_waveform_tables()
    assert len(tables) >= 1
    t = tables[0]
    assert t["waveformTable"] == "tset_gen_tp1"
    assert t["periodNs"] == 20.0
    assert t["frequencyMhz"] == 50.0
    assert t["captureEdgeSpacingNs"] == 0.5
    half = half_rate_reference(t)
    assert half["frequencyMhz"] == 25.0
    assert half["periodNs"] == 40.0


def test_format_matches_product_example():
    text = format_timing_violation_result(
        pattern_id="2210",
        fast_mhz=800,
        slow_mhz=400,
        capture_spacing_ns=1.25,
        near_minimum=True,
        multi_insertion=True,
    )
    assert "Pattern #2210 fails only at 800MHz timing set (passes at 400MHz)" in text
    assert "capture edge spacing per STIL = 1.25ns" in text
    assert "near minimum defined margin" in text


def test_kpi_count_and_row_shape(dataset):
    clear_timing_violation_caches()
    rows = build_timing_violation_results(dataset)
    assert count_timing_violations(dataset) == len(rows)
    assert len(rows) >= 1
    for row in rows:
        assert "fails only at" in row["result"]
        assert "capture edge spacing per STIL" in row["result"]
        assert row["atSpeedCorrelated"] is True
        assert row["fastFrequencyMhz"] == 50.0
        assert row["slowFrequencyMhz"] == 25.0
        assert row["captureEdgeSpacingNs"] == 0.5
        assert row["kind"] in ("setup", "hold", "anomaly", "timing")


def test_warm_cache(dataset):
    clear_timing_violation_caches()
    a = warm_timing_violation_cache(dataset)
    b = warm_timing_violation_cache(dataset)
    assert a == b
    assert a >= 1
