"""Pytest defaults — keep tests unauthenticated and non-production."""

import os

os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("REQUIRE_API_KEY", "false")
os.environ.setdefault("STARTUP_WARM_CACHES", "false")
os.environ.setdefault("AUTO_TRAIN_ON_STARTUP", "false")
