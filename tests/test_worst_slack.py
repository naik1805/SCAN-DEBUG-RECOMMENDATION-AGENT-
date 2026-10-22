"""Tests for Worst Slack (frequency margin proxy + ps)."""

from __future__ import annotations

import pytest

from src.data.dataset_builder import build_compiled_dataset
from src.data.timing_violations import clear_timing_violation_caches
from src.data.worst_slack import (
    build_worst_slack_results,
    clear_worst_slack_caches,
    format_worst_slack_result,
    frequency_margin_pct,
    warm_worst_slack_cache,
    worst_slack_kpi_value,
)


@pytest.fixture(scope="module")
def dataset():
    clear_timing_violation_caches()
    clear_worst_slack_caches()
    return build_compiled_dataset(write=False)


def test_frequency_margin_matches_product_example():
    # (800 − 650) / 800 = 18.75 → ~19
    assert frequency_margin_pct(800, 650) == 19


def test_format_with_and_without_slack_ps():
    text = format_worst_slack_result(
        fail_mhz=800,
        pass_mhz=650,
        margin_pct=19,
        worst_slack_ps=-47,
    )
    assert "Fails at 800MHz, passes at 650MHz — ~19% frequency margin proxy" in text
    assert "worst slack -47 ps" in text or "worst slack −47 ps" in text or "worst slack -47ps" in text.replace(
        " ", ""
    )
    # Normalize: our formatter uses "-47 ps"
    assert "worst slack" in text and "47" in text and "ps" in text

    no_ps = format_worst_slack_result(
        fail_mhz=800,
        pass_mhz=650,
        margin_pct=19,
        worst_slack_ps=None,
    )
    assert "worst slack" not in no_ps


def test_build_rows_and_kpi(dataset):
    clear_worst_slack_caches()
    rows = build_worst_slack_results(dataset)
    assert len(rows) >= 1
    value = worst_slack_kpi_value(dataset)
    assert "~" in value and "%" in value
    top = rows[0]
    assert "Fails at" in top["result"]
    assert "passes at" in top["result"]
    assert "frequency margin proxy" in top["result"]
    assert top["failFrequencyMhz"] == 50.0
    assert top["passFrequencyMhz"] == 25.0
    assert top["frequencyMarginPct"] == 50.0
    assert top["worstSlackPs"] is not None
    assert top["worstSlackPs"] < 0
    assert "worst slack" in top["result"]


def test_warm_cache(dataset):
    clear_worst_slack_caches()
    a = warm_worst_slack_cache(dataset)
    b = warm_worst_slack_cache(dataset)
    assert a == b
    assert a >= 1
