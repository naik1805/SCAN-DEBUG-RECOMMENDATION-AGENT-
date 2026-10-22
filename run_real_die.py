import os
import json
import re
import argparse
import sys
import httpx

from src.data.paths import DATA_DIR, diagnosis_html_path, FAILING_BITMAPS_PATH
from src.data.dataset_builder import _parse_diagnosis_html, clean_key
from src.config import get_settings

_settings = get_settings()
API_URL = os.getenv(
    "API_URL",
    f"http://{_settings.api_host}:{_settings.api_port}/recommend",
)

def clean_str(s):
    return clean_key(s)

def get_recommendation_for_die(lot_id, die_label):
    print(f"Analyzing Lot: {lot_id} | Die: {die_label}...")
    
    lot_clean = clean_str(lot_id)
    die_clean = clean_str(die_label)
    
    # 1. Locate and Parse Log File
    log_file_name = f"{die_label}.log" if not die_label.endswith(".log") else die_label
    # Find in lot directory
    # Map lot folder name (might be "lot 1", "lot2", etc.)
    log_path = None
    for folder in os.listdir(os.path.join(DATA_DIR, "failure logs")):
        if clean_str(folder) == lot_clean:
            possible_path = os.path.join(DATA_DIR, "failure logs", folder, log_file_name)
            if os.path.exists(possible_path):
                log_path = possible_path
                break
                
    if not log_path:
        print(f"Error: Log file for Lot {lot_id} and Die {die_label} not found.")
        sys.exit(1)
        
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_content = f.read()
        
    # Extract log headers
    defect_type = re.findall(r"DEFECT_TYPE\s*:\s*(\w+)", log_content)[0]
    fails_count = len(re.findall(r"STATUS\s*:\s*F", log_content))
    
    # 2. Parse HTML Tables (Breaks, Slacks)
    break_keys, slack_by_die, faults_by_die = _parse_diagnosis_html(diagnosis_html_path())
    die_key = f"{lot_clean}|{die_clean}"
    has_break = die_key in break_keys
    min_slack = slack_by_die.get(die_key, 9999.0)
                
    # 3. Pull coordinates from Bitmap JSON
    bitmap_path = FAILING_BITMAPS_PATH
    with open(bitmap_path, "r") as f:
        bitmap_json = json.load(f)
        
    coordinates = []
    for lot in bitmap_json.get("lots", []):
        if clean_str(lot.get("lot_id", "")) != lot_clean:
            continue
        for die in lot.get("dies", []):
            if clean_str(die.get("die_label", "")) == die_clean and die.get("status") == "fail":
                coordinates.append({"x": int(die.get("die_col") or 1), "y": int(die.get("die_row") or 1)})
                
    # 4. Construct Payload
    payload = {
        "failure_logs": {
            "mismatch_count": fails_count,
            "shifter_failure": has_break and (defect_type.upper() in ["SCRATCH", "RANDOM"]),
            "defect_type": defect_type
        },
        "diagnosis_results": {
            "suspected_chains": ["suspect_chain"] if has_break else [],
            "suspected_cells": [],
            "fault_models": [f"Slack {min_slack} ps"] if min_slack < 9999.0 else []
        },
        "failing_bitmaps": {
            "coordinates": coordinates
        }
    }
    
    # 5. Query Recommendation API
    try:
        r = httpx.post(API_URL, json=payload, timeout=5.0)
        if r.status_code != 200:
            print(f"Error querying API: {r.status_code} - {r.text}")
            sys.exit(1)
            
        rec = r.json()
        print("\n" + "=" * 60)
        print("SCAN DEBUG AGENT RECOMMENDATION REPORT")
        print("=" * 60)
        print(f"Lot / Die ID:         {lot_id} / {die_label}")
        print(f"Wafer Defect Shape:   {defect_type}")
        print(f"Mismatch Fails:       {fails_count}")
        print(f"Suspected Bitmaps:    {len(coordinates)} coordinates")
        print(f"Min Timing Slack:     {min_slack if min_slack < 9999.0 else 'N/A'} ps")
        print("-" * 60)
        print(f"RECOMMENDED ACTION:    {rec['recommended_action']}")
        print(f"MODEL CONFIDENCE:      {rec['confidence']*100:.2f}%")
        print(f"RECOMMENDATION WHY:    {rec['rationale']}")
        print("=" * 60 + "\n")
        
    except httpx.ConnectError:
        print("\n[Error]: Could not connect to API server. Make sure the FastAPI web server is running!")
        print("To start the server, run: uvicorn src.api.main:app --reload\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query recommendation agent for a specific Lot/Die log file.")
    parser.add_argument("--lot", required=True, help="Lot ID (e.g. LOT_1, LOT_6)")
    parser.add_argument("--die", required=True, help="Die Label (e.g. fail_die_1, fail_die_5)")
    
    args = parser.parse_args()
    get_recommendation_for_die(args.lot, args.die)
