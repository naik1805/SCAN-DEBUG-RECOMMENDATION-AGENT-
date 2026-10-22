import numpy as np
import re
from src.api.schemas import ScanAnalysisInput

STATE_DIM = 10

DEFECT_MAP = {
    "CENTER": 0.1,
    "NEAR_FULL": 0.2,
    "LOCAL": 0.3,
    "DONUT": 0.4,
    "EDGE_RING": 0.5,
    "EDGE_LOC": 0.6,
    "SCRATCH": 0.7,
    "RANDOM": 0.8,
    "NORMAL": 0.9
}

def build_state_vector(input_data: ScanAnalysisInput) -> np.ndarray:
    """
    Converts a ScanAnalysisInput into a normalized 10-dimensional numpy array state vector matching real data.
    
    Dimensions:
    0: Normalized mismatch count (fail_count / 100.0, capped at 1.0)
    1: Shifter failure flag (1.0 if shifter_failure is True, else 0.0)
    2: Suspected chains count normalized (count / 10.0, capped at 1.0)
    3: Suspected cells count normalized (count / 100.0, capped at 1.0)
    4: Timing fault flag (1.0 if fault model has transition, hold, or setup, else 0.0)
    5: Stuck-at fault flag (1.0 if fault model has stuck-at, else 0.0)
    6: Bitmap coordinate density (number of coordinates / 100.0, capped at 1.0)
    7: Bounding box area ratio (width * height / 10000.0, capped at 1.0)
    8: Defect Type encoding (using DEFECT_MAP, defaults to 0.9 for NORMAL)
    9: Min slack value normalized (mapped from -100 to 100 ps, default 1.0 for pass/no anomaly)
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)
    
    # 0. Mismatch count (typically 40 to 70 in real logs)
    state[0] = min(input_data.failure_logs.mismatch_count / 100.0, 1.0)
    
    # 1. Shifter failure flag
    state[1] = 1.0 if input_data.failure_logs.shifter_failure else 0.0
    
    # 2. Suspected chains
    state[2] = min(len(input_data.diagnosis_results.suspected_chains) / 10.0, 1.0)
    
    # 3. Suspected cells
    state[3] = min(len(input_data.diagnosis_results.suspected_cells) / 100.0, 1.0)
    
    # 4. Timing fault flag
    faults = [f.lower() for f in input_data.diagnosis_results.fault_models]
    has_timing = any(t in f for f in faults for t in ["transition", "hold", "setup", "timing", "delay"])
    state[4] = 1.0 if has_timing else 0.0
    
    # 5. Stuck-at fault flag
    has_stuck = any(s in f for f in faults for s in ["stuck", "stuck-at", "sa0", "sa1"])
    state[5] = 1.0 if has_stuck else 0.0
    
    # 6. Bitmap coordinate density
    state[6] = min(len(input_data.failing_bitmaps.coordinates) / 100.0, 1.0)
    
    # 7. Bounding box area ratio
    area = input_data.failing_bitmaps.bounding_box_width * input_data.failing_bitmaps.bounding_box_height
    state[7] = min(area / 10000.0, 1.0)
    
    # 8. Defect Type encoding
    defect_type = (input_data.failure_logs.defect_type or "NORMAL").upper()
    state[8] = DEFECT_MAP.get(defect_type, 0.9)
    
    # 9. Min Slack value
    # We parse slack values from historical cases or the patterns if provided.
    # To determine min slack, we look for setup/hold slack in the diagnostic results or cases:
    min_slack = 9999.0
    for fault in input_data.diagnosis_results.fault_models:
        # Check if there is a slack number like "-40 ps" in the text
        slack_match = re.search(r"(-?\d+(?:\.\d+)?)\s*ps", fault.lower())
        if slack_match:
            min_slack = min(min_slack, float(slack_match.group(1)))
            
    # Normalize slack between -100 and 100 ps
    if min_slack == 9999.0:
        state[9] = 1.0  # Safe/No timing issue
    else:
        # Clip slack to [-100, 100] and map to [0.0, 1.0]
        clipped = max(-100.0, min(100.0, min_slack))
        state[9] = (clipped + 100.0) / 200.0
        
    return state
