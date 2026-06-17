"""Tests for application lifespan behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.app import create_app
from src.api.lifespan import RuntimeSettings, load_runtime_settings, maybe_warmup_model


pytestmark = pytest.mark.anyio


class FakeModelManager:
    """Simple fake model manager used for warmup tests."""

    def __init__(self) -> None:
        """Store requested model sources."""
        self.requested_sources: list[Path] = []

    def get_model(self, model_source: str | Path) -> object:
        """Record model warmup requests."""
        resolved = Path(model_source)
        self.requested_sources.append(resolved)
        return object()


async def test_lifespan_sets_startup_and_shutdown_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure FastAPI lifespan performs startup work successfully."""
    monkeypatch.setenv("YOLO_STARTUP_VALIDATE_PATHS", "true")
    monkeypatch.setenv("YOLO_WARMUP_ENABLED", "false")

    app = create_app()

    async with app.router.lifespan_context(app):
        assert app.state.startup_completed is True
        assert app.state.model_warmup_performed is False
        assert app.state.runtime_settings.models_dir.exists()
        assert app.state.runtime_settings.outputs_dir.exists()


def test_maybe_warmup_model_loads_model_when_enabled() -> None:
    """Ensure optional model warmup loads the configured model into the manager."""
    settings = RuntimeSettings(
        startup_validate_paths=True,
        warmup_enabled=True,
        warmup_model_source="models/yolo11n.pt",
        warmup_skip_if_missing=False,
        models_dir=Path("models").resolve(),
        outputs_dir=Path("outputs").resolve(),
    )
    fake_manager = FakeModelManager()

    result = maybe_warmup_model(settings, model_manager=fake_manager)  # type: ignore[arg-type]

    assert result is True
    assert fake_manager.requested_sources
    assert fake_manager.requested_sources[0].name == "yolo11n.pt"


def test_load_runtime_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure runtime settings are driven by environment variables."""
    monkeypatch.setenv("YOLO_STARTUP_VALIDATE_PATHS", "false")
    monkeypatch.setenv("YOLO_WARMUP_ENABLED", "true")
    monkeypatch.setenv("YOLO_WARMUP_MODEL", "models/yolo11n.pt")
    monkeypatch.setenv("YOLO_WARMUP_SKIP_IF_MISSING", "false")
    monkeypatch.setenv("YOLO_MODELS_DIR", "models")
    monkeypatch.setenv("YOLO_OUTPUTS_DIR", "outputs")

    settings = load_runtime_settings()

    assert settings.startup_validate_paths is False
    assert settings.warmup_enabled is True
    assert settings.warmup_model_source == "models/yolo11n.pt"
    assert settings.warmup_skip_if_missing is False
