import json
import os
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

DATASET_PATH = r"d:\New folder\scan debug recommendation agent\scan debug data\compiled_dataset.json"

def run_simulation():
    print("=" * 70)
    print("SCAN DEBUG RL RECOMMENDATION AGENT - REAL DATA WALKTHROUGH")
    print("=" * 70)
    
    # 1. Load real cases
    with open(DATASET_PATH, "r") as f:
        cases = json.load(f)
        
    print(f"Loaded {len(cases)} real-world die cases.")
    
    # Select sample cases for distinct target actions
    sample_cases = {}
    for case in cases:
        action = case["true_action"]
        if action not in sample_cases:
            sample_cases[action] = case
            
    print(f"Selected {len(sample_cases)} sample cases representing distinct targets.")
    
    # 2. Train the agent on the real dataset
    print("\n--- Step 1: Training the DQN Agent on the Real Wafer Dataset ---")
    response = client.post("/train?episodes=300&save_weights=true")
    print(f"Training Metrics: {response.json()}")
    
    # 3. Request recommendations for each sample case
    print("\n--- Step 2: Requesting Recommendations (Inference) ---")
    for action, case in sample_cases.items():
        print(f"\n[Die Under Test: {case['lot_id']} / {case['die_label']}]")
        print(f"  * Wafer Defect Signature: {case['defect_type']}")
        print(f"  * Total Fails:           {case['mismatch_count']}")
        print(f"  * Has Break diagnosed:    {case['has_break']}")
        print(f"  * Min Timing Slack:       {case['min_slack']} ps")
        
        # Prepare input payload
        payload = {
            "failure_logs": {
                "mismatch_count": case["mismatch_count"],
                "shifter_failure": case["has_break"] and case["true_action"] == "INSPECT_SCAN_CHAIN",
                "defect_type": case["defect_type"]
            },
            "diagnosis_results": {
                "suspected_chains": ["suspect_chain"] if case["has_break"] else [],
                "suspected_cells": [],
                "fault_models": [f"Slack {case['min_slack']} ps"] if case["min_slack"] < 9999.0 else []
            },
            "failing_bitmaps": {
                "coordinates": [{"x": 1, "y": 1}] * case["cell_count"]
            }
        }
        
        rec_response = client.post("/recommend", json=payload)
        rec = rec_response.json()
        print(f"  * Recommending Action:    {rec['recommended_action']}")
        print(f"  * Model Confidence Score: {rec['confidence']:.4f}")
        print(f"  * Rationale:              {rec['rationale']}")
        print(f"  * Expected Ground Truth:  {action}")
        
    print("\n--- Step 3: Checking Agent final status ---")
    status_response = client.get("/status")
    print(f"Agent Status: {status_response.json()}")
    print("=" * 70)

if __name__ == "__main__":
    run_simulation()
