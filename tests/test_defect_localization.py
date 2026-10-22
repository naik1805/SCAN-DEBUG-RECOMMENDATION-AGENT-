"""Tests for Defect Localization confidence KPI."""

from __future__ import annotations

import pytest

from src.data.dataset_builder import build_compiled_dataset
from src.data.defect_localization import (
    average_defect_localization_confidence,
    build_defect_localization_results,
    clear_defect_localization_caches,
    compute_localization_confidence,
    format_defect_localization_result,
    get_defect_localization_summary,
    warm_defect_localization_cache,
)


@pytest.fixture(scope="module")
def dataset():
    clear_defect_localization_caches()
    return build_compiled_dataset(write=False)


def test_format_product_string():
    text = format_defect_localization_result(
        confidence_pct=87,
        net_id="N166",
        neighbor_from="U164",
        neighbor_to="U166",
        wafer_x=130.2,
        wafer_y=84.1,
        diagnosis_rank=1,
        consistent=4,
        total=4,
        hist_count=5,
        priority="High",
    )
    assert text.startswith("Defect localization confidence: 87%")
    assert "Net N166 (U164→U166)" in text
    assert "at wafer (130.2, 84.1)" in text
    assert "rank 1" in text
    assert "consistency 4/4" in text
    assert "PFA precedent 5" in text
    assert "priority High" in text


def test_compute_confidence_bounded():
    score, pct = compute_localization_confidence(
        diagnosis_conf_pct=74.42,
        consistency_ratio=1.0,
        power_induced_ruled_out=False,
        historical_match_count=5,
        xy_available=True,
        debug_priority="High",
    )
    assert 0 < score <= 0.99
    assert 0 < pct <= 99
    assert pct == round(score * 100)


def test_build_rows(dataset):
    clear_defect_localization_caches()
    rows = build_defect_localization_results(dataset)
    summary = get_defect_localization_summary(dataset)
    assert len(rows) >= 1
    assert "Defect localization confidence:" in rows[0]["result"]
    assert rows[0]["confidencePct"] >= 1
    assert average_defect_localization_confidence(dataset) == summary["kpiValue"]
    assert str(summary["kpiValue"]).endswith("%")


def test_warm_cache(dataset):
    clear_defect_localization_caches()
    a = warm_defect_localization_cache(dataset)
    b = warm_defect_localization_cache(dataset)
    assert a == b
    assert a >= 1
