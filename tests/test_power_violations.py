"""Tests for Power Violations (IR_DROP_MV / THERMAL_C thresholds)."""

from __future__ import annotations

import pytest

from src.data.dataset_builder import build_compiled_dataset
from src.data.power_violations import (
    IR_DROP_MV_THRESHOLD,
    THERMAL_C_THRESHOLD,
    build_power_violation_results,
    clear_power_violation_caches,
    count_power_violations,
    format_power_violation_result,
    power_violations_kpi_value,
    scan_log_power_metrics,
    warm_power_violation_cache,
)
from src.data.paths import FAILURE_LOGS_DIR
import os


@pytest.fixture(scope="module")
def dataset():
    clear_power_violation_caches()
    return build_compiled_dataset(write=False)


def test_format_matches_product_example():
    text = format_power_violation_result(
        pattern_id="4521",
        ir_drop_mv=20,
        thermal_c=50,
        status="P",
        ir_threshold=15,
        thermal_threshold=55,
    )
    assert "Pattern #4521: IR_DROP = 20mV (threshold 15mV)" in text
    assert "flagged despite PASS status" in text


def test_format_thermal_when_only_thermal_exceeds():
    text = format_power_violation_result(
        pattern_id="10",
        ir_drop_mv=10,
        thermal_c=65,
        status="P",
        ir_threshold=25,
        thermal_threshold=60,
    )
    assert "THERMAL = 65°C" in text
    assert "threshold 60°C" in text
    assert "despite PASS" in text


def test_scan_log_captures_fail_status():
    path = os.path.join(FAILURE_LOGS_DIR, "LOT_1_Center", "fail_die_1.log")
    if not os.path.exists(path):
        pytest.skip("sample log missing")
    rows = scan_log_power_metrics(path)
    assert len(rows) == 1000
    p8 = next(r for r in rows if r["patternId"] == "8")
    assert p8["irDropMv"] == 68.0
    assert p8["status"] == "F"


def test_build_and_count(dataset):
    clear_power_violation_caches()
    rows = build_power_violation_results(dataset)
    assert count_power_violations(dataset) == len(rows)
    assert len(rows) >= 1
    assert len(rows) <= 1000
    assert len({r["patternId"] for r in rows}) == len(rows)
    assert IR_DROP_MV_THRESHOLD == 25.0
    assert THERMAL_C_THRESHOLD == 60.0
    top = rows[0]
    assert "Pattern #" in top["result"]
    assert "threshold" in top["result"]
    assert top["irThresholdMv"] == 25.0
    # Marginal PASS patterns are flagged too (may not be the top severity rows)
    assert any(r.get("flaggedDespitePass") for r in rows)
    assert any(not r.get("flaggedDespitePass") for r in rows)


def test_kpi_value_is_unique_patterns_out_of_1000(dataset):
    value = power_violations_kpi_value(dataset)
    assert value.endswith("/1000")
    n = int(value.split("/", 1)[0])
    assert 1 <= n <= 1000

def test_warm_cache(dataset):
    clear_power_violation_caches()
    a = warm_power_violation_cache(dataset)
    b = warm_power_violation_cache(dataset)
    assert a == b
    assert a >= 1
