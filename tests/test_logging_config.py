"""Tests for structured logging configuration."""

from __future__ import annotations

from io import StringIO

import httpx
import pytest
from loguru import logger

from src.api.app import create_app
from src.utils.logging_config import LoggingSettings, configure_logging, load_logging_settings


pytestmark = pytest.mark.anyio


def test_configure_logging_writes_human_readable_logs() -> None:
    """Ensure logging configuration writes messages to the configured sink."""
    sink = StringIO()
    settings = LoggingSettings(level="INFO", serialize=False, log_file=None)

    configure_logging(settings, sink=sink)
    logger.bind(request_id="req-123").info("hello logging")

    output = sink.getvalue()
    assert "hello logging" in output
    assert "req-123" in output


def test_load_logging_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure logging settings are loaded from environment variables."""
    monkeypatch.setenv("YOLO_LOG_LEVEL", "debug")
    monkeypatch.setenv("YOLO_LOG_JSON", "true")
    monkeypatch.setenv("YOLO_LOG_FILE", "logs/test.log")

    settings = load_logging_settings()

    assert settings.level == "DEBUG"
    assert settings.serialize is True
    assert str(settings.log_file).endswith("logs/test.log")


async def test_request_logging_middleware_adds_request_id_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure request logging middleware adds a stable request ID header."""
    monkeypatch.setenv("YOLO_WARMUP_ENABLED", "false")
    transport = httpx.ASGITransport(app=create_app())

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"]
