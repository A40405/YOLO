"""Structured logging configuration utilities for the API runtime."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LoggingSettings:
    """Environment-driven logging settings for the API runtime."""

    level: str = "INFO"
    serialize: bool = False
    log_file: Path | None = None


def load_logging_settings(env: dict[str, str] | None = None) -> LoggingSettings:
    """Load structured logging settings from environment variables."""
    environment = env or os.environ
    level = environment.get("YOLO_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    serialize = _parse_bool(environment.get("YOLO_LOG_JSON", "false"))
    log_file_value = environment.get("YOLO_LOG_FILE", "logs/app.log").strip()
    log_file = _resolve_log_path(log_file_value) if log_file_value else None
    return LoggingSettings(level=level, serialize=serialize, log_file=log_file)


def configure_logging(
    settings: LoggingSettings | None = None,
    *,
    sink: TextIO | None = None,
) -> LoggingSettings:
    """Configure loguru for structured application logging."""
    resolved_settings = settings or load_logging_settings()
    logger.remove()

    log_sink = sink or sys.stderr
    if resolved_settings.serialize:
        logger.add(
            log_sink,
            level=resolved_settings.level,
            serialize=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        logger.add(
            log_sink,
            level=resolved_settings.level,
            backtrace=False,
            diagnose=False,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | "
                "{extra[request_id]} | {name}:{function}:{line} | {message}"
            ),
        )

    if resolved_settings.log_file is not None:
        resolved_settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            resolved_settings.log_file,
            level=resolved_settings.level,
            serialize=resolved_settings.serialize,
            backtrace=False,
            diagnose=False,
            rotation="10 MB",
            retention=5,
        )

    logger.configure(extra={"request_id": "-"})
    logger.info("Logging configured")
    return resolved_settings


def _resolve_log_path(log_file_value: str) -> Path:
    """Resolve a configured log file path relative to the project root."""
    candidate = Path(log_file_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _parse_bool(value: str) -> bool:
    """Parse a boolean environment variable."""
    return value.strip().lower() in {"1", "true", "yes", "on"}
