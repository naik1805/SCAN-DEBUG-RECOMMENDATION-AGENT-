"""
Central runtime configuration — all deploy-time values come from the environment.

Load order:
  1) process env
  2) optional project-root `.env` (does not override existing env)
  3) documented defaults for local development only
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_dotenv(path: str) -> None:
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_dotenv(os.path.join(_project_root(), ".env"))


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    return value


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _split_csv(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    return Settings()


class Settings:
    """Immutable snapshot of env-backed settings (cached via get_settings)."""

    def __init__(self) -> None:
        root = _project_root()

        self.project_root = root
        self.api_host = _env("API_HOST", "127.0.0.1") or "127.0.0.1"
        self.api_port = _env_int("API_PORT", 8005)
        self.ui_base_url = (
            _env("UI_BASE_URL")
            or f"http://{_env('UI_HOST', '127.0.0.1')}:{_env_int('UI_PORT', 3001)}"
        )
        self.ui_dashboard_path = _env(
            "UI_DASHBOARD_PATH", "/dashboard/recommendation-analysis"
        ) or "/dashboard/recommendation-analysis"

        origins = _split_csv(_env("CORS_ORIGINS"))
        if not origins:
            # Derive from UI_BASE_URL so LAN IPs are never hard-coded in source.
            origins = [self.ui_base_url.rstrip("/")]
            # Common local aliases when UI is on localhost
            host = _env("UI_HOST", "127.0.0.1") or "127.0.0.1"
            port = _env_int("UI_PORT", 3001)
            for h in {host, "localhost", "127.0.0.1"}:
                origins.append(f"http://{h}:{port}")
            origins = list(dict.fromkeys(origins))
        self.cors_origins = origins
        self.cors_allow_credentials = _env_bool("CORS_ALLOW_CREDENTIALS", True)

        data_dir = _env("SCAN_DEBUG_DATA_DIR")
        self.data_dir = (
            os.path.abspath(data_dir)
            if data_dir
            else os.path.join(root, "scan debug data")
        )
        stil_name = _env("STIL_FILENAME", "Production_SCAN_stuck_at_1000pat.stil")
        self.stil_path = (
            os.path.abspath(_env("STIL_PATH"))
            if _env("STIL_PATH")
            else os.path.join(self.data_dir, stil_name or "Production_SCAN_stuck_at_1000pat.stil")
        )
        weights = _env("MODEL_WEIGHTS_PATH")
        self.model_weights_path = (
            os.path.abspath(weights) if weights else os.path.join(root, "model_weights.pth")
        )

        self.auto_train_on_startup = _env_bool("AUTO_TRAIN_ON_STARTUP", True)
        self.auto_train_episodes = _env_int("AUTO_TRAIN_EPISODES", 500)
        self.dashboard_cache_ttl_sec = _env_float("DASHBOARD_CACHE_TTL_SEC", 90.0)
        self.top_recommendations_limit = _env_int("TOP_RECOMMENDATIONS_LIMIT", 20)
        self.fetch_timeout_ms = _env_int("DASHBOARD_FETCH_TIMEOUT_MS", 120_000)

        self.ir_drop_mv_threshold = _env_float("IR_DROP_MV_THRESHOLD", 25.0)
        self.thermal_c_threshold = _env_float("THERMAL_C_THRESHOLD", 60.0)
        self.ir_normal_threshold_mv = _env_float("IR_NORMAL_THRESHOLD_MV", 15.0)
        self.scan_chain_high_similarity = _env_float("SCAN_CHAIN_HIGH_SIMILARITY", 0.78)

        self.power_workspace_rows = _env_int("POWER_WORKSPACE_ROWS", 400)
        self.defect_workspace_rows = _env_int("DEFECT_WORKSPACE_ROWS", 40)

        # Production / security
        self.app_env = (_env("APP_ENV", "development") or "development").strip().lower()
        self.is_production = self.app_env in ("production", "prod")
        self.log_level = (_env("LOG_LEVEL", "INFO" if self.is_production else "DEBUG") or "INFO").upper()
        self.log_json = _env_bool("LOG_JSON", self.is_production)
        self.api_keys = _split_csv(_env("API_KEYS")) or (
            [_env("API_KEY")] if _env("API_KEY") else []
        )
        self.require_api_key = _env_bool(
            "REQUIRE_API_KEY", self.is_production and bool(self.api_keys)
        )
        self.rate_limit_per_minute = _env_int("RATE_LIMIT_PER_MINUTE", 120)
        self.rate_limit_train_per_minute = _env_int("RATE_LIMIT_TRAIN_PER_MINUTE", 5)
        self.disable_openapi = _env_bool("DISABLE_OPENAPI", self.is_production)
        self.trusted_hosts = _split_csv(_env("TRUSTED_HOSTS"))
        self.max_train_episodes = _env_int("MAX_TRAIN_EPISODES", 1000)
        self.startup_warm_caches = _env_bool("STARTUP_WARM_CACHES", True)

        # Supervised ML recommendations (RL remains separate for self-learning)
        self.recommendation_source = (
            _env("RECOMMENDATION_SOURCE", "ml") or "ml"
        ).strip().lower()
        ml_path = _env("ML_MODEL_PATH")
        self.ml_model_path = (
            os.path.abspath(ml_path)
            if ml_path
            else os.path.join(self.data_dir, "ml_action_recommender.joblib")
        )
        self.ml_auto_train = _env_bool("ML_AUTO_TRAIN", True)
        self.ml_use_for_api_recommend = _env_bool("ML_USE_FOR_API_RECOMMEND", True)

    @property
    def ui_dashboard_url(self) -> str:
        return f"{self.ui_base_url.rstrip('/')}{self.ui_dashboard_path}"

    def validate_production(self) -> None:
        """Fail fast when production is misconfigured."""
        if not self.is_production:
            return
        if self.require_api_key and not self.api_keys:
            raise RuntimeError(
                "APP_ENV=production requires API_KEY or API_KEYS when REQUIRE_API_KEY is enabled."
            )


# Convenience module-level accessors used by older import sites
def data_dir() -> str:
    return get_settings().data_dir


def model_weights_path() -> str:
    return get_settings().model_weights_path


def stil_path() -> str:
    return get_settings().stil_path
