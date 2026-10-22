# Production deployment

## Checklist

1. Copy `.env.example` → `.env` and set **`API_KEY`** (long random secret).
2. Copy `frontend/.env.example` → `frontend/.env.local` and set **`API_KEY`** (same value, server-side only — never `NEXT_PUBLIC_*`).
3. Set `APP_ENV=production`, `REQUIRE_API_KEY=true`, `LOG_JSON=true`.
4. Set `CORS_ORIGINS` to your UI origin(s).
5. Mount scan data read-only; version `model_weights.pth` per release.

## Run with Docker

```bash
cp .env.example .env
# edit API_KEY, CORS_ORIGINS, UI_BASE_URL

cp frontend/.env.example frontend/.env.local
# set API_KEY + API_PROXY_TARGET=http://api:8005

docker compose up --build
```

- UI: `http://localhost:3001`
- API health: `http://localhost:8005/health`
- API ready: `http://localhost:8005/ready`

## Security

| Feature | Env |
|---------|-----|
| API key auth | `API_KEY` / `API_KEYS`, `REQUIRE_API_KEY` |
| Rate limits | `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_TRAIN_PER_MINUTE` |
| Disable OpenAPI | `DISABLE_OPENAPI=true` (default in production) |
| Structured logs | `LOG_JSON=true` |

The browser never receives the API key — Next.js route `scan-debug-api/[...path]` attaches `X-API-Key` server-side.

## Protected endpoints (production)

Requires `X-API-Key` or `Authorization: Bearer <key>`:

- `POST /train`, `POST /feedback`, `POST /recommend`
- All `/api/v1/*` reads
- `/status`, `/analyze-die`

Public: `/health`, `/ready`

## Model governance

- Pin `model_weights.pth` per deployment.
- Set `AUTO_TRAIN_ON_STARTUP=false` in production if weights are pre-validated.
- Cap training: `MAX_TRAIN_EPISODES`.

## Local development

```bash
APP_ENV=development REQUIRE_API_KEY=false python -m uvicorn src.api.main:app --reload
cd frontend && npm run dev
```
