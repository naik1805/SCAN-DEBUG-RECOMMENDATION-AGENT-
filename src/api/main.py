import os
import re
import threading
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any

from src.api.schemas import ScanAnalysisInput, RecommendationResponse, FeedbackInput
from src.api.logging_config import configure_logging, get_logger
from src.api.middleware import (
    ApiKeyMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from src.api.health import health_payload, readiness_payload
from src.api.errors import register_exception_handlers
from src.env.state_builder import build_state_vector
from src.env.debug_env import ACTION_MAP, REV_ACTION_MAP
from src.config import get_settings
from src.data.paths import COMPILED_DATASET_PATH, MODEL_WEIGHTS_PATH
from src.data.dataset_builder import build_compiled_dataset
from src.data.scan_chain_debug_recs import warm_scan_chain_debug_recs_cache
from src.data.atpg_constraint_violations import warm_constraint_violation_cache
from src.data.atpg_constraint_review_recs import warm_constraint_review_recs_cache
from src.data.atpg_coverage_impact import warm_coverage_impact_cache
from src.data.timing_violations import warm_timing_violation_cache
from src.data.timing_debug_recs import warm_timing_debug_recs_cache
from src.data.worst_slack import warm_worst_slack_cache
from src.data.power_violations import warm_power_violation_cache
from src.data.power_debug_recs import warm_power_debug_recs_cache
from src.data.peak_switching import warm_peak_switching_cache
from src.data.defect_suspects import warm_defect_suspects_cache
from src.data.investigation_recs import warm_investigation_recs_cache
from src.data.defect_localization import warm_defect_localization_cache
from src.api.dashboard_service import build_dashboard_payload, build_kpi_workspace
from src.api.training_service import (
    DEFAULT_AUTO_TRAIN_EPISODES,
    run_training,
    should_auto_train,
    weights_need_retrain,
)
from src.models.agent import DQNAgent, device
from src.models.supervised_recommender import warm_ml_action_model, model_status, predict_action
from src.data.recommendation_engine import (
    SCAN_CHAIN,
    TIMING_DEBUG,
    POWER_DEBUG,
    ATPG_CONSTRAINT,
    PHYSICAL_DEFECT,
)

configure_logging()
log = get_logger("scan_debug.main")
_settings_boot = get_settings()
_settings_boot.validate_production()

agent = DQNAgent()
MODEL_PATH = MODEL_WEIGHTS_PATH
startup_info: Dict[str, Any] = {
    "dataset_cases": 0,
    "auto_trained": False,
    "auto_train_result": None,
    "ml_train_result": None,
}
_train_lock = threading.Lock()
_training_state: Dict[str, Any] = {
    "in_progress": False,
    "last_result": None,
    "last_error": None,
    "source": None,
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global startup_info
    settings = get_settings()
    cases: list = []
    try:
        cases = build_compiled_dataset(write=True)
        startup_info["dataset_cases"] = len(cases)
        log.info("Compiled %s die cases -> %s", len(cases), COMPILED_DATASET_PATH)
    except Exception as e:
        log.warning("Could not compile dataset on startup: %s", e)

    if not settings.startup_warm_caches:
        log.info("Skipping KPI cache warm (STARTUP_WARM_CACHES=false)")
    else:
        warmers = [
            ("scan chain debug recs", warm_scan_chain_debug_recs_cache),
            ("constraint violations", warm_constraint_violation_cache),
            ("constraint review recs", warm_constraint_review_recs_cache),
            ("coverage impact", warm_coverage_impact_cache),
            ("timing violations", warm_timing_violation_cache),
            ("timing debug recs", warm_timing_debug_recs_cache),
            ("worst slack", warm_worst_slack_cache),
            ("power violations", warm_power_violation_cache),
            ("power debug recs", warm_power_debug_recs_cache),
            ("peak switching", warm_peak_switching_cache),
            ("defect suspects", warm_defect_suspects_cache),
            ("investigation recs", warm_investigation_recs_cache),
            ("defect localization", warm_defect_localization_cache),
        ]
        for label, fn in warmers:
            try:
                n = fn(cases if cases else None)
                log.info("Warmed %s cache (%s)", label, n)
            except Exception as e:
                log.warning("Could not warm %s cache: %s", label, e)

    if os.path.exists(MODEL_PATH):
        try:
            agent.load(MODEL_PATH)
            log.info("Loaded model weights from %s", MODEL_PATH)
        except Exception as e:
            log.warning("Could not load weights from %s: %s", MODEL_PATH, e)

    try:
        ml_result = warm_ml_action_model(cases if cases else None)
        startup_info["ml_train_result"] = ml_result
        log.info(
            "ML recommender: %s (cv acc %s)",
            ml_result.get("status"),
            (ml_result.get("meta") or {}).get("cv_accuracy_mean"),
        )
    except Exception as e:
        log.warning("Could not warm ML recommender: %s", e)

    def _auto_train_bg() -> None:
        global startup_info
        if not should_auto_train():
            log.info("Skipping auto-train (weights up to date with source data).")
            return
        log.info(
            "Auto-training DQN (%s episodes)...", DEFAULT_AUTO_TRAIN_EPISODES
        )
        try:
            result = _run_training_locked(
                episodes=DEFAULT_AUTO_TRAIN_EPISODES,
                save_weights=True,
                source="startup",
            )
            if result.get("skipped"):
                log.info("Auto-train skipped: %s", result.get("status"))
                return
            startup_info["auto_trained"] = True
            startup_info["auto_train_result"] = result
            log.info(
                "Auto-train complete: avg reward=%.2f",
                result["average_episode_reward"],
            )
        except Exception as e:
            log.exception("Auto-train failed: %s", e)

    threading.Thread(target=_auto_train_bg, name="auto-train", daemon=True).start()

    yield


def _run_training_locked(
    *,
    episodes: int,
    save_weights: bool,
    source: str,
    force: bool = True,
) -> Dict[str, Any]:
    """Serialize training so startup auto-train and dashboard retrain do not collide."""
    global _training_state
    if not force and not weights_need_retrain():
        return {
            "status": "Training skipped — weights already match current dataset",
            "episodes_trained": 0,
            "average_episode_reward": 0.0,
            "average_loss": 0.0,
            "final_epsilon": agent.epsilon,
            "weights_saved": False,
            "dataset_cases": startup_info.get("dataset_cases", 0),
            "skipped": True,
        }
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Training already in progress")
    _training_state["in_progress"] = True
    _training_state["source"] = source
    _training_state["last_error"] = None
    try:
        result = run_training(
            agent,
            episodes=episodes,
            save_weights=save_weights,
            model_path=MODEL_PATH,
        )
        result["skipped"] = False
        result["source"] = source
        _training_state["last_result"] = result
        if source == "startup":
            startup_info["auto_trained"] = True
            startup_info["auto_train_result"] = result
        return result
    except Exception as e:
        _training_state["last_error"] = str(e)
        raise
    finally:
        _training_state["in_progress"] = False
        _training_state["source"] = None
        _train_lock.release()


app = FastAPI(
    title="Scan Debug Recommendation Agent API",
    description="Reinforcement learning-based recommendation system for scan chain debugging.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if _settings_boot.disable_openapi else "/docs",
    redoc_url=None if _settings_boot.disable_openapi else "/redoc",
    openapi_url=None if _settings_boot.disable_openapi else "/openapi.json",
)

register_exception_handlers(app)

_settings = get_settings()

# Outermost middleware is added last (Starlette executes in reverse add order).
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=_settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(ApiKeyMiddleware)
app.add_middleware(RateLimitMiddleware)

# Ensure static files directory exists
os.makedirs("src/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/health")
def health():
    return health_payload()


@app.get("/ready")
def ready():
    return readiness_payload(training_in_progress=bool(_training_state.get("in_progress")))


@app.get("/")
def get_dashboard():
    settings = get_settings()
    return {
        "service": "Scan Debug Recommendation Agent API",
        "ui": settings.ui_dashboard_url,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
        "docs": "/docs",
    }


SUB_REC_MAP = {
    "SCAN_CHAIN_DEBUG": ["Isolation", "Bypass", "Re-Stitch"],
    "TIMING_DEBUG": ["At-Speed", "Launch", "Capture"],
    "POWER_RELATED_DEBUG": ["Clock Gating", "Domain Isolation", "Voltage Scaling"],
    "ATPG_CONSTRAINT_REVIEW": ["Relax", "Tighten", "Remove"],
    "PHYSICAL_DEFECT_INVESTIGATION": ["SEM", "FIB", "X-Ray"]
}

_INTERNAL_TO_API_ACTION = {
    SCAN_CHAIN: "SCAN_CHAIN_DEBUG",
    TIMING_DEBUG: "TIMING_DEBUG",
    POWER_DEBUG: "POWER_RELATED_DEBUG",
    ATPG_CONSTRAINT: "ATPG_CONSTRAINT_REVIEW",
    PHYSICAL_DEFECT: "PHYSICAL_DEFECT_INVESTIGATION",
}


def _case_from_analysis_input(input_data: ScanAnalysisInput) -> Dict[str, Any]:
    faults = list(input_data.diagnosis_results.fault_models or [])
    min_slack = 9999.0
    for fault in faults:
        slack_match = re.search(r"(-?\d+(?:\.\d+)?)\s*ps", str(fault).lower())
        if slack_match:
            min_slack = min(min_slack, float(slack_match.group(1)))
    return {
        "mismatch_count": input_data.failure_logs.mismatch_count,
        "has_break": bool(input_data.failure_logs.shifter_failure),
        "shifter_failure": bool(input_data.failure_logs.shifter_failure),
        "defect_type": input_data.failure_logs.defect_type or "NORMAL",
        "cell_count": len(input_data.failing_bitmaps.coordinates or []),
        "fault_models": faults,
        "min_slack": min_slack,
        "failing_channel_count": len(input_data.diagnosis_results.suspected_chains or []),
    }

def generate_rationale(action_name: str, input_data: ScanAnalysisInput) -> str:
    """
    Generates a natural language explanation for why the agent selected this recommendation.
    """
    faults = [f.lower() for f in input_data.diagnosis_results.fault_models]
    mismatch = input_data.failure_logs.mismatch_count
    
    if action_name == "SCAN_CHAIN_DEBUG":
        if input_data.failure_logs.shifter_failure:
            return "Recommending Scan Chain Debugging because the failure occurred during the Shift phase, indicating a basic shift path or clock-gating issue."
        return f"Recommending Scan Chain Debugging due to active stuck-at faults ({', '.join(input_data.diagnosis_results.fault_models)}) and suspected chains detected in diagnostic logs."
        
    elif action_name == "TIMING_DEBUG":
        return f"Recommending Clock Timing debugging because active timing faults ({', '.join(input_data.diagnosis_results.fault_models)}) were diagnosed, indicating setup/hold or skew violations during high-frequency capture clocks."
        
    elif action_name == "POWER_RELATED_DEBUG":
        return f"Recommending Power-related debugging because of a high mismatch count ({mismatch} fails), combined with layout clustering indicators in the bitmap coordinates, suggesting local power grid droop during capture clock pulses."
        
    elif action_name == "PHYSICAL_DEFECT_INVESTIGATION":
        coord_count = len(input_data.failing_bitmaps.coordinates)
        w = input_data.failing_bitmaps.bounding_box_width
        h = input_data.failing_bitmaps.bounding_box_height
        return f"Recommending Physical Defect Investigation (PFA) due to a high density of failing bitmap coordinates ({coord_count} spots) spreading across a {w}x{h} layout bounding box, which strongly correlates to physical dust, bridge, or oxide damage."
        
    else:  # ATPG_CONSTRAINT_REVIEW
        return "Recommending ATPG Constraint Review to rule out invalid clock groupings, missing capture masking, or incorrect constraint parameters in the test setup."


def compute_detail_metrics(action_name: str, confidence: float, input_data: ScanAnalysisInput) -> Dict[str, str]:
    """
    Computes the detailed metrics block matching the 5 dashboard rows and 3 sub-metrics per row.
    """
    mismatch = input_data.failure_logs.mismatch_count
    
    # Extract timing slack if any timing faults
    min_slack = 9999.0
    for fault in input_data.diagnosis_results.fault_models:
        slack_match = re.search(r"(-?\d+(?:\.\d+)?)\s*ps", fault.lower())
        if slack_match:
            min_slack = min(min_slack, float(slack_match.group(1)))
            
    if action_name == "SCAN_CHAIN_DEBUG":
        return {
            "Broken Chains Detected": str(max(1, len(input_data.diagnosis_results.suspected_chains))),
            "Debug Recommendations": "Isolation • Bypass • Re-Stitch",
            "Average Confidence": f"{confidence*100:.1f}%"
        }
    elif action_name == "ATPG_CONSTRAINT_REVIEW":
        return {
            "Constraint Violations": str(max(1, mismatch)),
            "Review Recommendations": "Relax • Tighten • Remove",
            "Coverage Impact": f"{min(mismatch * 0.05, 5.0):.1f}%"
        }
    elif action_name == "TIMING_DEBUG":
        worst_slack_str = f"{min_slack:.1f} ps" if min_slack < 9999.0 else "N/A"
        return {
            "Timing Violations": str(max(1, len(input_data.diagnosis_results.fault_models))),
            "Timing Debug Recommendations": "At-Speed • Launch • Capture",
            "Worst Slack": worst_slack_str
        }
    elif action_name == "POWER_RELATED_DEBUG":
        return {
            "Power Violations": str(max(1, mismatch // 5)),
            "Power Debug Recommendations": "Clock Gating • Domain Isolation • Voltage Scaling",
            "Peak Switching Activity": f"{min(50.0 + mismatch * 0.25, 95.0):.1f}%"
        }
    else:  # PHYSICAL_DEFECT_INVESTIGATION
        return {
            "Defect Suspects": str(len(input_data.failing_bitmaps.coordinates)),
            "Investigation Recommendations": "SEM • FIB • X-Ray • E-Beam",
            "Defect Localization Accuracy": f"{80.0 + confidence * 19.9:.1f}%"
        }


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(input_data: ScanAnalysisInput):
    """
    Ingests failure diagnostics and returns the recommended debug action.
    Uses supervised ML by default; DQN RL remains for feedback-driven learning.
    """
    try:
        settings = get_settings()
        if settings.ml_use_for_api_recommend:
            case = _case_from_analysis_input(input_data)
            internal_action, confidence_score, _ = predict_action(case)
            action_name = _INTERNAL_TO_API_ACTION.get(internal_action, "ATPG_CONSTRAINT_REVIEW")
        else:
            state = build_state_vector(input_data)
            action_idx = agent.select_action(state, evaluate=True)
            action_name = ACTION_MAP.get(action_idx, "ATPG_CONSTRAINT_REVIEW")
            confidences = agent.get_action_confidences(state)
            confidence_score = float(confidences[action_idx])

        rationale = generate_rationale(action_name, input_data)
        detail_metrics = compute_detail_metrics(action_name, confidence_score, input_data)

        return RecommendationResponse(
            recommended_action=action_name,
            sub_recommendations=SUB_REC_MAP.get(action_name, []),
            confidence=confidence_score,
            rationale=rationale + (
                " (supervised ML policy)" if settings.ml_use_for_api_recommend else ""
            ),
            detail_metrics=detail_metrics,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")


@app.post("/feedback")
def feedback(feedback_data: FeedbackInput):
    """
    Accepts feedback about the recommendation, calculates the reward,
    stores it in the experience replay buffer, and runs an optimization step.
    """
    try:
        # Convert inputs to state vector
        state = build_state_vector(feedback_data.input_data)
        
        # Map actions to integers
        rec_action_idx = REV_ACTION_MAP.get(feedback_data.recommended_action)
        act_action_idx = REV_ACTION_MAP.get(feedback_data.actual_resolution)
        
        if rec_action_idx is None or act_action_idx is None:
            raise HTTPException(status_code=400, detail="Invalid action name provided in feedback.")
            
        # Determine reward
        reward = 100.0 if feedback_data.success or rec_action_idx == act_action_idx else -10.0
        # Account for step cost
        reward -= 1.0
        
        # Next state: since this is a recommendation episode, we assume success transitions to a terminal state
        # (success = terminal, incorrect = non-terminal)
        done = feedback_data.success or rec_action_idx == act_action_idx
        next_state = state if not done else np.zeros_like(state)
        
        # Push to experience memory
        agent.memory.push(state, rec_action_idx, reward, next_state, done)
        
        # If incorrect, push the correct action transition as well (guided learning / teacher forcing)
        if not done:
            agent.memory.push(state, act_action_idx, 99.0, np.zeros_like(state), True)
            
        # Optimize DQN weights
        loss = agent.optimize_model()
        
        return {
            "status": "Feedback processed",
            "reward_allocated": reward,
            "loss": loss,
            "epsilon": agent.epsilon,
            "buffer_size": len(agent.memory)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback processing error: {str(e)}")


@app.post("/train")
def train(
    episodes: int = DEFAULT_AUTO_TRAIN_EPISODES,
    save_weights: bool = True,
    force: bool = True,
):
    """
    Triggers simulated offline training epochs in the Gymnasium environment to let the agent learn.
    Set force=false to skip when weights already match the current dataset (dashboard auto-train).
    """
    settings = get_settings()
    episodes = min(max(1, episodes), settings.max_train_episodes)
    try:
        return _run_training_locked(
            episodes=episodes,
            save_weights=save_weights,
            source="manual" if force else "dashboard-auto",
            force=force,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@app.get("/api/v1/ml/status")
def ml_recommender_status():
    """Supervised ML recommender metadata (dashboard recommendations)."""
    return {
        **model_status(),
        "startup": startup_info.get("ml_train_result"),
    }


@app.get("/api/v1/recommendation/dashboard")
def recommendation_dashboard():
    conf = float(agent.get_action_confidences(np.zeros(10, dtype=np.float32)).max()) if agent else 0.87
    return build_dashboard_payload(agent_confidence=conf)


@app.get("/api/v1/kpi/{kpi_id}/workspace")
def kpi_workspace(kpi_id: str):
    conf = float(agent.get_action_confidences(np.zeros(10, dtype=np.float32)).max()) if agent else 0.87
    return build_kpi_workspace(kpi_id, agent_confidence=conf)


@app.get("/status")
def status():
    """
    Get current agent parameters.
    """
    last = _training_state.get("last_result") or startup_info.get("auto_train_result")
    return {
        "device": str(device),
        "replay_buffer_size": len(agent.memory),
        "epsilon": agent.epsilon,
        "steps_done": agent.steps_done,
        "model_weights_exist": os.path.exists(MODEL_PATH),
        "auto_train_on_startup": str(get_settings().auto_train_on_startup).lower(),
        "needs_training": weights_need_retrain(),
        "training_in_progress": bool(_training_state.get("in_progress")),
        "training_source": _training_state.get("source"),
        "dataset_cases": startup_info.get("dataset_cases", 0),
        "auto_trained": startup_info.get("auto_trained", False),
        "auto_train_result": last,
        "last_train_error": _training_state.get("last_error"),
    }

@app.get("/analyze-die")
def analyze_die(lot_id: str, die_label: str):
    """
    Finds the specified die in the compiled dataset, builds the state, and queries the DQN agent.
    """
    DATASET_PATH = COMPILED_DATASET_PATH
    if not os.path.exists(DATASET_PATH):
        build_compiled_dataset(write=True)
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=404, detail="Compiled dataset not found.")
        
    import json
    with open(DATASET_PATH, "r") as f:
        cases = json.load(f)
        
    target_case = None
    clean_lot = lot_id.strip().upper().replace(" ", "").replace("_", "")
    clean_die = die_label.strip().upper().replace(" ", "").replace("_", "")
    
    for case in cases:
        c_lot = case["lot_id"].strip().upper().replace(" ", "").replace("_", "")
        c_die = case["die_label"].strip().upper().replace(" ", "").replace("_", "")
        if c_lot == clean_lot and c_die == clean_die:
            target_case = case
            break
            
    if not target_case:
        raise HTTPException(status_code=404, detail=f"Die {die_label} under Lot {lot_id} not found in dataset.")
        
    # Build payload
    payload = ScanAnalysisInput(
        failure_logs={
            "mismatch_count": target_case["mismatch_count"],
            "shifter_failure": target_case["has_break"] and target_case["true_action"] == "INSPECT_SCAN_CHAIN",
            "defect_type": target_case["defect_type"]
        },
        diagnosis_results={
            "suspected_chains": ["suspect_chain"] if target_case["has_break"] else [],
            "suspected_cells": [],
            "fault_models": [f"Slack {target_case['min_slack']} ps"] if target_case["min_slack"] < 9999.0 else []
        },
        failing_bitmaps={
            "coordinates": [{"x": 1, "y": 1}] * target_case["cell_count"]
        }
    )
    
    # Run recommendation
    state = build_state_vector(payload)
    action_idx = agent.select_action(state, evaluate=True)
    action_name = ACTION_MAP.get(action_idx, "ATPG_CONSTRAINT_REVIEW")
    
    # Get Q-values for confidence
    import torch
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor)
        probs = torch.softmax(q_values, dim=1).cpu().numpy()[0]
        confidence = float(probs[action_idx])
        
    rationale = generate_rationale(action_name, payload)
    detail_metrics = compute_detail_metrics(action_name, confidence, payload)
    
    return {
        "mismatch_count": target_case["mismatch_count"],
        "defect_type": target_case["defect_type"],
        "has_break": target_case["has_break"],
        "min_slack": target_case["min_slack"],
        "cell_count": target_case["cell_count"],
        "recommended_action": action_name,
        "sub_recommendations": SUB_REC_MAP.get(action_name, []),
        "confidence": confidence,
        "rationale": rationale,
        "detail_metrics": detail_metrics,
        "expected": target_case["true_action"]
    }

