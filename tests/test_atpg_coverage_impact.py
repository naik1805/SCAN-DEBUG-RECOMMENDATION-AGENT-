"""Tests for ATPG coverage impact (failing-pattern share proxy)."""

from __future__ import annotations

import pytest

from src.data.atpg_coverage_impact import (
    build_coverage_impact_results,
    clear_coverage_impact_caches,
    coverage_impact_kpi_value,
    format_coverage_impact_result,
    warm_coverage_impact_cache,
)
from src.data.atpg_constraint_violations import clear_constraint_violation_caches
from src.data.dataset_builder import build_compiled_dataset


@pytest.fixture(scope="module")
def fail_cases():
    clear_constraint_violation_caches()
    clear_coverage_impact_caches()
    dataset = build_compiled_dataset(write=False)
    return [c for c in dataset if str(c.get("source_file", "")).startswith("fail")]


def test_kpi_value_is_overall_share(fail_cases):
    rows = build_coverage_impact_results(fail_cases)
    overall = next(r for r in rows if r.get("scope") == "overall")
    value = coverage_impact_kpi_value(fail_cases)
    assert value.startswith("~")
    assert value.endswith("%")
    assert "ATPG constraints" in overall["result"]
    cats = [r for r in rows if r.get("scope") == "category"]
    assert len(cats) == 3
    assert {c["constraintCategory"] for c in cats} == {"reset", "scan_enable", "clock"}


def test_format_coverage_impact_result():
    text = format_coverage_impact_result(
        associated=40,
        total=650,
        pct=6.0,
        signature="RESET_N=1 × SPI0_SCSN",
    )
    assert "~6%" in text
    assert "(40/650)" in text
    assert "estimate only" in text
    assert "RESET_N=1 × SPI0_SCSN" in text


def test_build_coverage_impact_rows(fail_cases):
    rows = build_coverage_impact_results(fail_cases)
    assert len(rows) >= 15
    sig_rows = [r for r in rows if r.get("scope") == "signature"]
    assert len(sig_rows) >= 15
    for row in sig_rows:
        assert "estimate only" in row["result"]
        assert row["totalFailingPatterns"] >= row["associatedPatterns"]
        assert 0 <= float(row["coverageImpactPct"]) <= 100


def test_warm_cache(fail_cases):
    clear_coverage_impact_caches()
    a = warm_coverage_impact_cache(fail_cases)
    b = warm_coverage_impact_cache(fail_cases)
    assert a == b
    assert a >= 15
