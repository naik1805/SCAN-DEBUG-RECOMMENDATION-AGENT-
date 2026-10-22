"""Production middleware: request id, security headers, API key auth, rate limits."""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Optional, Set, Tuple

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings

log = logging.getLogger("scan_debug.api")

# Paths always public (no API key)
_PUBLIC_PATHS: Set[str] = {"/health", "/ready", "/healthz", "/readyz"}

# Prefixes public for GET only (dashboard reads)
_PUBLIC_GET_PREFIXES: Tuple[str, ...] = (
    "/static",
    "/docs",
    "/redoc",
    "/openapi.json",
)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _extract_api_key(request: Request) -> Optional[str]:
    header = request.headers.get("x-api-key")
    if header:
        return header.strip()
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _path_requires_auth(method: str, path: str) -> bool:
    settings = get_settings()
    if not settings.require_api_key:
        return False
    if path in _PUBLIC_PATHS:
        return False
    if method == "GET" and path == "/":
        return False
    if method == "GET" and any(path.startswith(p) for p in _PUBLIC_GET_PREFIXES):
        return False
    # Sensitive writes always require key when auth is enabled
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        return True
    # Production: protect all API routes and agent endpoints
    if settings.is_production and (
        path.startswith("/api/") or path in ("/status", "/recommend", "/analyze-die", "/feedback", "/train")
    ):
        return True
    return False


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "%s %s -> %s (%.0fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "client_ip": _client_ip(request),
            },
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-XSS-Protection", "0")
        if get_settings().is_production:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()
        path = request.url.path
        if not _path_requires_auth(request.method, path):
            return await call_next(request)

        provided = _extract_api_key(request)
        if not provided or not any(secrets.compare_digest(provided, k) for k in settings.api_keys):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter (per client IP + route class)."""

    def __init__(self, app) -> None:
        super().__init__(app)
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def _limit_for(self, path: str) -> int:
        settings = get_settings()
        if path == "/train" or path.startswith("/train"):
            return settings.rate_limit_train_per_minute
        return settings.rate_limit_per_minute

    def _bucket_key(self, request: Request) -> str:
        path = request.url.path
        if path.startswith("/api/v1/kpi"):
            path = "/api/v1/kpi"
        return f"{_client_ip(request)}:{path}"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        settings = get_settings()
        if settings.rate_limit_per_minute <= 0:
            return await call_next(request)

        now = time.time()
        window = 60.0
        key = self._bucket_key(request)
        limit = self._limit_for(request.url.path)
        q = self._hits[key]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Retry later."},
                headers={"Retry-After": "60"},
            )
        q.append(now)
        return await call_next(request)
