"""Tests for Peak Switching (IR_DROP_MV proxy)."""

from __future__ import annotations

import pytest

from src.data.dataset_builder import build_compiled_dataset
from src.data.peak_switching import (
    build_peak_switching_results,
    clear_peak_switching_caches,
    format_peak_switching_result,
    get_peak_switching_summary,
    peak_switching_kpi_value,
    warm_peak_switching_cache,
)
from src.data.power_violations import clear_power_violation_caches


@pytest.fixture(scope="module")
def dataset():
    clear_power_violation_caches()
    clear_peak_switching_caches()
    return build_compiled_dataset(write=False)


def test_format_matches_product_example():
    text = format_peak_switching_result(
        peak_ir_mv=20.0,
        pattern_id="4521",
        avg_ir_mv=8.0,
    )
    assert text == "Peak IR-drop: 20mV at Pattern #4521 (vs. avg 8mV across run)"


def test_build_peak_and_avg(dataset):
    clear_peak_switching_caches()
    rows = build_peak_switching_results(dataset)
    summary = get_peak_switching_summary(dataset)
    assert len(rows) >= 1
    assert summary.get("peakIrDropMv") is not None
    assert summary.get("avgIrDropMv") is not None
    assert summary["peakIrDropMv"] >= summary["avgIrDropMv"]
    assert rows[0].get("isPeak") is True
    assert "Peak IR-drop:" in rows[0]["result"]
    assert "at Pattern #" in rows[0]["result"]
    assert "vs. avg" in rows[0]["result"]
    assert "across run" in rows[0]["result"]
    assert peak_switching_kpi_value(dataset) == summary["kpiValue"]
    # Rows sorted by IR descending
    irs = [float(r["irDropMv"]) for r in rows]
    assert irs == sorted(irs, reverse=True)


def test_warm_cache(dataset):
    clear_peak_switching_caches()
    a = warm_peak_switching_cache(dataset)
    b = warm_peak_switching_cache(dataset)
    assert a == b
    assert a >= 1
