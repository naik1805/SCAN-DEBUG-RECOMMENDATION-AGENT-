import os

from src.config import get_settings

_settings = get_settings()

PROJECT_ROOT = _settings.project_root
DATA_DIR = _settings.data_dir
FAILURE_LOGS_DIR = os.path.join(DATA_DIR, "failure logs")
COMPILED_DATASET_PATH = os.path.join(DATA_DIR, "compiled_dataset.json")
FAILING_BITMAPS_PATH = os.path.join(DATA_DIR, "failing_bitmaps.json")
CONSTRAINT_VIOLATIONS_PATH = os.path.join(DATA_DIR, "constraint_violations.json")
CONSTRAINT_REVIEW_RECS_PATH = os.path.join(DATA_DIR, "constraint_review_recommendations.json")
COVERAGE_IMPACT_PATH = os.path.join(DATA_DIR, "coverage_impact.json")
SCAN_CHAIN_DEBUG_RECS_PATH = os.path.join(DATA_DIR, "scan_chain_debug_recommendations.json")
TIMING_VIOLATIONS_PATH = os.path.join(DATA_DIR, "timing_violations.json")
TIMING_DEBUG_RECS_PATH = os.path.join(DATA_DIR, "timing_debug_recommendations.json")
WORST_SLACK_PATH = os.path.join(DATA_DIR, "worst_slack.json")
POWER_VIOLATIONS_PATH = os.path.join(DATA_DIR, "power_violations.json")
POWER_DEBUG_RECS_PATH = os.path.join(DATA_DIR, "power_debug_recommendations.json")
PEAK_SWITCHING_PATH = os.path.join(DATA_DIR, "peak_switching.json")
DEFECT_SUSPECTS_PATH = os.path.join(DATA_DIR, "defect_suspects.json")
INVESTIGATION_RECS_PATH = os.path.join(DATA_DIR, "investigation_recommendations.json")
DEFECT_LOCALIZATION_PATH = os.path.join(DATA_DIR, "defect_localization.json")
HISTORICAL_CASES_PATH = os.path.join(DATA_DIR, "historical_debug_cases.json")
MODEL_WEIGHTS_PATH = _settings.model_weights_path
STIL_PATH = _settings.stil_path


def diagnosis_html_path() -> str:
    """Return the diagnosis report HTML (supports renamed export files)."""
    preferred = os.path.join(DATA_DIR, "diagnosis result.html")
    if not os.path.isdir(DATA_DIR):
        return preferred
    if os.path.exists(preferred):
        return preferred
    for name in os.listdir(DATA_DIR):
        lower = name.lower()
        if lower.endswith(".html") and ("diagnosis" in lower or "scd-fr" in lower):
            return os.path.join(DATA_DIR, name)
    for name in os.listdir(DATA_DIR):
        if name.lower().endswith(".html"):
            return os.path.join(DATA_DIR, name)
    return preferred
