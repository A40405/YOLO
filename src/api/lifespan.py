"""FastAPI lifespan support for startup validation and warmup."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from loguru import logger

from src.core.model_manager import ModelManager
from src.utils.logging_config import configure_logging, load_logging_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RuntimeSettings:
    """Environment-driven API runtime settings."""

    startup_validate_paths: bool
    warmup_enabled: bool
    warmup_model_source: str
    warmup_skip_if_missing: bool
    models_dir: Path
    outputs_dir: Path


def load_runtime_settings(env: dict[str, str] | None = None) -> RuntimeSettings:
    """Load production runtime settings from environment variables."""
    environment = env or os.environ
    return RuntimeSettings(
        startup_validate_paths=_parse_bool(environment.get("YOLO_STARTUP_VALIDATE_PATHS", "true")),
        warmup_enabled=_parse_bool(environment.get("YOLO_WARMUP_ENABLED", "false")),
        warmup_model_source=environment.get("YOLO_WARMUP_MODEL", "models/yolo11n.pt").strip() or "models/yolo11n.pt",
        warmup_skip_if_missing=_parse_bool(environment.get("YOLO_WARMUP_SKIP_IF_MISSING", "true")),
        models_dir=_resolve_project_path(environment.get("YOLO_MODELS_DIR", "models")),
        outputs_dir=_resolve_project_path(environment.get("YOLO_OUTPUTS_DIR", "outputs")),
    )


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup validation, optional model warmup, and shutdown logging."""
    logging_settings = configure_logging(load_logging_settings())
    runtime_settings = load_runtime_settings()
    app.state.logging_settings = logging_settings
    app.state.runtime_settings = runtime_settings
    app.state.startup_completed = False
    app.state.model_warmup_performed = False

    validate_startup(runtime_settings)
    warmup_performed = maybe_warmup_model(runtime_settings)
    app.state.model_warmup_performed = warmup_performed
    app.state.startup_completed = True
    logger.info("Application startup completed")

    try:
        yield
    finally:
        logger.info("Application shutdown completed")


def validate_startup(settings: RuntimeSettings) -> None:
    """Validate required project directories and warmup settings."""
    if not settings.startup_validate_paths:
        logger.info("Startup path validation disabled")
        return

    _ensure_directory_exists(settings.models_dir, label="Models directory")
    _ensure_directory_exists(settings.outputs_dir, label="Outputs directory")

    if settings.warmup_enabled:
        warmup_model_path = _resolve_project_path(settings.warmup_model_source)
        if not warmup_model_path.exists() and not settings.warmup_skip_if_missing:
            raise FileNotFoundError(f"Warmup model not found: {warmup_model_path}")

    logger.info("Startup validation completed")


def maybe_warmup_model(settings: RuntimeSettings, model_manager: ModelManager | None = None) -> bool:
    """Optionally warm the model cache during application startup."""
    if not settings.warmup_enabled:
        logger.info("Model warmup disabled")
        return False

    resolved_model_path = _resolve_project_path(settings.warmup_model_source)
    if not resolved_model_path.exists():
        if settings.warmup_skip_if_missing:
            logger.warning("Skipping model warmup because model is missing: {}", resolved_model_path)
            return False
        raise FileNotFoundError(f"Warmup model not found: {resolved_model_path}")

    manager = model_manager or ModelManager()
    manager.get_model(resolved_model_path)
    logger.info("Model warmup completed for {}", resolved_model_path)
    return True


def _resolve_project_path(path_value: str) -> Path:
    """Resolve a path relative to the project root when needed."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _ensure_directory_exists(path_value: Path, *, label: str) -> None:
    """Ensure a required startup directory exists."""
    if not path_value.exists() or not path_value.is_dir():
        raise FileNotFoundError(f"{label} not found: {path_value}")


def _parse_bool(value: str) -> bool:
    """Parse a boolean environment variable."""
    return value.strip().lower() in {"1", "true", "yes", "on"}
