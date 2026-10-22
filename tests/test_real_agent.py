import os
import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.api.schemas import ScanAnalysisInput
from src.env.state_builder import build_state_vector
from src.env.debug_env import ScanDebugEnv, ACTION_MAP
from src.api.main import app

from src.data.paths import COMPILED_DATASET_PATH

DATASET_PATH = COMPILED_DATASET_PATH

client = TestClient(app)

def load_real_cases():
    with open(DATASET_PATH, "r") as f:
        return json.load(f)

def test_real_state_builder():
    """Verify that state vectorizer works with the structure of real-world compiled dies."""
    cases = load_real_cases()
    assert len(cases) > 0
    
    # Construct a valid ScanAnalysisInput from a compiled record
    case = cases[0]
    input_data = ScanAnalysisInput(
        failure_logs={
            "mismatch_count": case["mismatch_count"],
            "shifter_failure": case["has_break"] and case["true_action"] == "INSPECT_SCAN_CHAIN",
            "defect_type": case["defect_type"]
        },
        diagnosis_results={
            "suspected_chains": ["suspect_chain"] if case["has_break"] else [],
            "suspected_cells": [],
            "fault_models": [f"Slack {case['min_slack']} ps"] if case["min_slack"] < 9999.0 else []
        },
        failing_bitmaps={
            "coordinates": [{"x": 1, "y": 1}] * case["cell_count"]
        }
    )
    
    state = build_state_vector(input_data)
    assert isinstance(state, np.ndarray)
    assert state.shape == (10,)
    assert state[0] == pytest.approx(case["mismatch_count"] / 100.0)


def test_real_gymnasium_env():
    """Verify Gymnasium transitions with real dataset cases."""
    env = ScanDebugEnv()
    obs, info = env.reset()
    
    assert obs.shape == (10,)
    assert info["target_action"] in ACTION_MAP.values()
    
    # Step correct action
    action = info["target_action_idx"]
    next_obs, reward, done, truncated, step_info = env.step(action)
    assert reward == 99.0
    assert done
    assert step_info["correct"] is True


def test_api_endpoints_real():
    """Verify recommendation, feedback, and training endpoints with real data payloads."""
    cases = load_real_cases()
    case = cases[0]
    
    payload = {
        "failure_logs": {
            "mismatch_count": case["mismatch_count"],
            "shifter_failure": case["has_break"] and case["true_action"] == "INSPECT_SCAN_CHAIN",
            "defect_type": case["defect_type"]
        },
        "diagnosis_results": {
            "suspected_chains": ["chain1"] if case["has_break"] else [],
            "suspected_cells": [],
            "fault_models": [f"Timing Slack: {case['min_slack']} ps"] if case["min_slack"] < 9999.0 else []
        },
        "failing_bitmaps": {
            "coordinates": [{"x": 10, "y": 20}] * case["cell_count"]
        }
    }
    
    # Status
    response = client.get("/status")
    assert response.status_code == 200
    
    # Recommend
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert "recommended_action" in res_data
    assert "confidence" in res_data
    
    # Feedback
    feedback_payload = {
        "input_data": payload,
        "recommended_action": res_data["recommended_action"],
        "actual_resolution": case["true_action"],
        "success": res_data["recommended_action"] == case["true_action"]
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 200
    
    # Train (Real training loop)
    response = client.post("/train?episodes=10")
    assert response.status_code == 200
    train_data = response.json()
    assert train_data["episodes_trained"] == 10
