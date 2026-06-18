FROM ghcr.io/astral-sh/uv:0.8.15 AS uv

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN python -m venv /app/.venv \
    && uv export --no-dev --no-sources --format requirements.txt --output-file requirements.txt \
    && uv pip sync --python /app/.venv/bin/python --torch-backend cpu requirements.txt

COPY src ./src

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    UVICORN_WORKERS=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY pyproject.toml ./pyproject.toml

RUN mkdir -p /app/models /app/outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail "http://127.0.0.1:${API_PORT}/api/v1/health" || exit 1

CMD ["sh", "-c", "uvicorn src.api.app:app --host ${API_HOST} --port ${API_PORT} --workers ${UVICORN_WORKERS}"]
