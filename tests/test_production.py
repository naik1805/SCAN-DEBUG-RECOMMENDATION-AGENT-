import os

import pytest
from fastapi.testclient import TestClient

from src.api.middleware import _path_requires_auth


def test_path_requires_auth_production():
    assert _path_requires_auth("POST", "/train") is False  # dev default from conftest

    import src.config

    src.config.get_settings.cache_clear()
    os.environ["APP_ENV"] = "production"
    os.environ["REQUIRE_API_KEY"] = "true"
    os.environ["API_KEY"] = "secret"
    src.config.get_settings.cache_clear()

    assert _path_requires_auth("POST", "/train") is True
    assert _path_requires_auth("GET", "/health") is False
    assert _path_requires_auth("GET", "/api/v1/recommendation/dashboard") is True

    os.environ["APP_ENV"] = "development"
    os.environ["REQUIRE_API_KEY"] = "false"
    src.config.get_settings.cache_clear()


def test_health_and_request_id():
    from src.api.main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID")


def test_ready_shape():
    from src.api.main import app

    client = TestClient(app)
    r = client.get("/ready")
    body = r.json()
    assert "checks" in body
    assert "compiled_dataset" in body["checks"]
