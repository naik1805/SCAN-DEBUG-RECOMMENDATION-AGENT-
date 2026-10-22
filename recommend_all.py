import os
import json
import httpx
import sys

from src.data.paths import COMPILED_DATASET_PATH
from src.config import get_settings

DATASET_PATH = COMPILED_DATASET_PATH
_settings = get_settings()
API_URL = os.getenv(
    "API_URL",
    f"http://{_settings.api_host}:{_settings.api_port}/recommend",
)

def recommend_all_dies():
    # 1. Load the compiled dataset containing all 45 die cases
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Compiled dataset not found at {DATASET_PATH}.")
        sys.exit(1)
        
    with open(DATASET_PATH, "r") as f:
        cases = json.load(f)
        
    print(f"Loaded {len(cases)} die cases from your dataset. Querying API server...")
    
    results = []
    correct_count = 0
    
    # 2. Iterate through all cases and query the agent
    for case in cases:
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
        
        # Map old db expected actions to new UI actions
        db_to_ui = {
            "INSPECT_SCAN_CHAIN": "SCAN_CHAIN_DEBUG",
            "REVIEW_CAPTURE_CLOCK_TIMING": "TIMING_DEBUG",
            "CHECK_IR_DROP_DURING_CAPTURE": "POWER_RELATED_DEBUG",
            "REVIEW_ATPG_CONSTRAINTS": "ATPG_CONSTRAINT_REVIEW",
            "INVESTIGATE_PHYSICAL_DEFECT": "PHYSICAL_DEFECT_INVESTIGATION"
        }
        expected_ui = db_to_ui.get(case["true_action"], "ATPG_CONSTRAINT_REVIEW")
        
        try:
            r = httpx.post(API_URL, json=payload, timeout=5.0)
            if r.status_code != 200:
                print(f"Error querying API for {case['lot_id']}/{case['die_label']}: {r.text}")
                continue
                
            rec = r.json()
            is_correct = rec["recommended_action"] == expected_ui
            if is_correct:
                correct_count += 1
                
            results.append({
                "lot_id": case["lot_id"],
                "die_label": case["die_label"],
                "defect_type": case["defect_type"],
                "slack": case["min_slack"],
                "recommended": rec["recommended_action"],
                "confidence": rec["confidence"],
                "expected": expected_ui,
                "correct": "YES" if is_correct else "NO"
            })
        except httpx.ConnectError:
            print("\n[Error]: Could not connect to API server. Make sure the FastAPI web server is running!")
            print("To start the server, run: uvicorn src.api.main:app --reload\n")
            sys.exit(1)
            
    # 3. Print a beautiful aligned table of results
    print("\n" + "=" * 115)
    print(f"{'LOT ID':<10} | {'DIE LABEL':<12} | {'DEFECT SHAPE':<14} | {'SLACK':<8} | {'RECOMMENDED ACTION':<30} | {'CONFIDENCE':<10} | {'MATCH?'}")
    print("=" * 115)
    
    for r in results:
        slack_str = f"{r['slack']:.1f}" if r['slack'] < 9999.0 else "N/A"
        conf_str = f"{r['confidence']*100:.1f}%"
        print(f"{r['lot_id']:<10} | {r['die_label']:<12} | {r['defect_type']:<14} | {slack_str:<8} | {r['recommended']:<30} | {conf_str:<10} | {r['correct']}")
        
    print("=" * 115)
    print(f"Total Dies Analyzed: {len(results)}")
    print(f"Model Recommendations Match Expert Ground Truth: {correct_count} / {len(results)} ({correct_count/len(results)*100:.1f}%)")
    print("=" * 115 + "\n")
    
    # Save the output report
    report_path = "all_recommendations_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full detailed report to: {report_path}")

if __name__ == "__main__":
    recommend_all_dies()
