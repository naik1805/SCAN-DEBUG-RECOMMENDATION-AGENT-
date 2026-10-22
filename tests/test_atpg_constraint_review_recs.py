"""Tests for ATPG constraint review recommendations."""

from __future__ import annotations

import pytest

from src.data.atpg_constraint_review_recs import (
    build_constraint_review_recommendations,
    clear_constraint_review_recs_caches,
    count_constraint_review_recommendations,
    format_clock_review,
    format_reset_review,
    format_scan_enable_review,
    warm_constraint_review_recs_cache,
)
from src.data.atpg_constraint_violations import clear_constraint_violation_caches
from src.data.dataset_builder import build_compiled_dataset


@pytest.fixture(scope="module")
def fail_cases():
    clear_constraint_violation_caches()
    clear_constraint_review_recs_caches()
    dataset = build_compiled_dataset(write=False)
    return [c for c in dataset if str(c.get("source_file", "")).startswith("fail")]


def test_format_templates_match_category_language():
    se = format_scan_enable_review(
        pin_token="TEST=1",
        procedure="ext_capture_7",
        fanout="SPI0_SCSN",
        pattern_count=24,
        chains_hit=24,
        total_chains=24,
        start_cycle=3,
        hist_count=2,
        hist_resolution="relaxing to don't-care during capture",
    )
    assert "Suspected Scan Enable timing issue" in se
    assert "verify SE pulse alignment" in se
    assert "2 historical cases" in se

    clk = format_clock_review(
        pin_token="XI=0",
        procedure="ext_capture_7",
        clock_domain="ETH_RXCLK",
        fanout="ETH_TXCLK",
        pattern_count=40,
        chains_hit=8,
        total_chains=24,
        hist_count=2,
        hist_resolution="correcting capture clock pulse / gating constraint",
    )
    assert "Suspected Clock constraint issue" in clk
    assert "ETH_RXCLK" in clk
    assert "clock gating/pulse-generation" in clk

    rst = format_reset_review(
        pin_token="RESET_N=1",
        procedure="ext_capture_7",
        fanout="SPI0_IO2_WP",
        pattern_count=40,
        flop_count=6,
        reset_domain="RST_CORE",
        hist_count=2,
        hist_resolution="relaxing to don't-care during capture",
    )
    assert "Suspected Reset constraint issue" in rst
    assert "don't-care before capture edge" in rst
    assert "RST_CORE" in rst


def test_build_review_recs_has_all_categories(fail_cases):
    clear_constraint_review_recs_caches()
    rows = build_constraint_review_recommendations(fail_cases)
    assert len(rows) >= 15
    cats = {r["constraintCategory"] for r in rows}
    assert "reset" in cats
    assert "scan_enable" in cats
    assert "clock" in cats
    assert all(r.get("result") for r in rows)
    assert any(r.get("historicalMatchCount", 0) >= 1 for r in rows)


def test_count_matches_rows(fail_cases):
    clear_constraint_review_recs_caches()
    rows = build_constraint_review_recommendations(fail_cases)
    assert count_constraint_review_recommendations(fail_cases) == len(rows)


def test_warm_cache(fail_cases):
    clear_constraint_review_recs_caches()
    a = warm_constraint_review_recs_cache(fail_cases)
    b = warm_constraint_review_recs_cache(fail_cases)
    assert a == b
    assert a >= 15
