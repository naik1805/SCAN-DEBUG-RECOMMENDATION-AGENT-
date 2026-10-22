"""Tests for Defect Suspects KPI."""

from __future__ import annotations

import pytest

from src.data.dataset_builder import build_compiled_dataset
from src.data.defect_suspects import (
    build_defect_suspect_results,
    clear_defect_suspects_caches,
    count_defect_suspects,
    format_defect_suspect_clause,
    format_defect_suspects_recommendation,
    get_defect_suspect_summary,
    warm_defect_suspects_cache,
)


@pytest.fixture(scope="module")
def dataset():
    clear_defect_suspects_caches()
    return build_compiled_dataset(write=False)


def test_format_matches_product_example():
    c1 = format_defect_suspect_clause(
        net_id="N4521",
        diagnosis_rank=1,
        consistent=11,
        total=12,
        neighbor_from="U890",
        neighbor_to="U912",
        include_neighbors=True,
    )
    c2 = format_defect_suspect_clause(
        net_id="N4487",
        diagnosis_rank=2,
        consistent=6,
        total=12,
        include_neighbors=False,
    )
    text = format_defect_suspects_recommendation([c1, c2])
    assert (
        text
        == "Net N4521 (U890→U912) — diagnosis rank 1, consistent with 11/12 failing patterns; "
        "Net N4487 — rank 2, consistent with 6/12"
    )


def test_build_rows(dataset):
    clear_defect_suspects_caches()
    rows = build_defect_suspect_results(dataset)
    summary = get_defect_suspect_summary(dataset)
    assert count_defect_suspects(dataset) == int(summary["count"])
    assert len(rows) >= 1
    assert rows[0]["diagnosisRank"] == 1
    assert "Net N" in rows[0]["result"]
    assert "diagnosis rank 1" in rows[0]["result"]
    assert "consistent with" in rows[0]["result"]
    assert rows[0]["consistentPatterns"] <= rows[0]["totalFailingPatterns"]
    assert "Net N" in (summary.get("result") or "")
    assert summary.get("kpiValue")


def test_warm_cache(dataset):
    clear_defect_suspects_caches()
    a = warm_defect_suspects_cache(dataset)
    b = warm_defect_suspects_cache(dataset)
    assert a == b
    assert a >= 1
