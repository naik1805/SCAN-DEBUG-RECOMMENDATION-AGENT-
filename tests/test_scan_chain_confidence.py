"""Regression tests for scan chain confidence scoring."""

from __future__ import annotations

import math

import pytest

from src.api.dashboard_service import SCAN_CHAIN, _cases_for_action, build_dashboard_payload, build_kpi_workspace
from src.data.dataset_builder import build_compiled_dataset
from src.data.scan_chain_confidence import (
    average_scan_chain_confidence,
    build_scan_chain_confidence_results,
    clear_fr002_cache,
    compute_scan_chain_confidence,
    pattern_consistency_stats,
)


@pytest.fixture(scope="module")
def dataset():
    clear_fr002_cache()
    return build_compiled_dataset(write=False)


@pytest.fixture(scope="module")
def broken_cases(dataset):
    return [c for c in dataset if c.get("has_break")]


def test_all_broken_cases_score_without_error(broken_cases):
    assert len(broken_cases) == 44
    for case in broken_cases:
        row = compute_scan_chain_confidence(case)
        assert 0 <= row["confidencePct"] <= 99
        assert 0.0 <= row["confidenceScore"] <= 0.99
        assert not math.isnan(row["confidenceScore"])
        assert row["patternTotal"] >= 1
        assert 0 <= row["patternConsistent"] <= row["patternTotal"]
        assert row["ambiguityGroup"] >= 1
        assert row["result"]


def test_kpi_average_matches_manual_mean(dataset):
    scan_cases = _cases_for_action(dataset, SCAN_CHAIN)
    assert len(scan_cases) == 10
    scores = [compute_scan_chain_confidence(c)["confidenceScore"] for c in scan_cases]
    expected = f"{round(sum(scores) / len(scores) * 100)}%"
    assert average_scan_chain_confidence(scan_cases) == expected


def test_dashboard_kpi_matches_engine(dataset):
    payload = build_dashboard_payload(agent_confidence=0.5)
    kpi = next(k for k in payload["kpis"] if k["id"] == "avg_ai_confidence")
    scan_cases = _cases_for_action(dataset, SCAN_CHAIN)
    assert kpi["value"] == average_scan_chain_confidence(scan_cases)


def test_confidence_workspace_returns_rows(dataset):
    workspace = build_kpi_workspace("avg_ai_confidence", agent_confidence=0.5)
    rows = workspace["diagnosisResults"]
    assert workspace["layout"] == "scan_chain_confidence_clean"
    assert len(rows) == 44
    assert all("confidencePct" in r for r in rows)
    assert rows[0]["rank"] == 1


def test_empty_cases_return_zero_percent():
    assert average_scan_chain_confidence([]) == "0%"
    assert build_scan_chain_confidence_results([]) == []


def test_pattern_ratio_never_divide_by_zero():
    row = compute_scan_chain_confidence(
        {
            "chain_name": "channel1",
            "candidate_bit": 72,
            "fail_count": 0,
            "has_break": True,
        }
    )
    assert row["patternTotal"] >= 1
    assert not math.isnan(row["confidenceScore"])


def test_fr002_match_used_for_known_case(dataset):
    case = next(
        c
        for c in dataset
        if c.get("lot_id") == "LOT_8" and c.get("chain_name") == "channel1" and c.get("has_break")
    )
    consistent, total, ratio = pattern_consistency_stats(case)
    assert total <= 10
    assert consistent <= total
    assert ratio > 0.5
