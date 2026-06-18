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


async def test_lifespan_sets_startup_and_shutdown_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure FastAPI lifespan performs startup work successfully."""
    models_dir = (tmp_path / "models").resolve()
    outputs_dir = (tmp_path / "outputs").resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("YOLO_STARTUP_VALIDATE_PATHS", "true")
    monkeypatch.setenv("YOLO_WARMUP_ENABLED", "false")
    monkeypatch.setenv("YOLO_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("YOLO_OUTPUTS_DIR", str(outputs_dir))

    app = create_app()

    async with app.router.lifespan_context(app):
        assert app.state.startup_completed is True
        assert app.state.model_warmup_performed is False
        assert app.state.runtime_settings.models_dir == models_dir
        assert app.state.runtime_settings.outputs_dir == outputs_dir


def test_maybe_warmup_model_loads_model_when_enabled(tmp_path: Path) -> None:
    """Ensure optional model warmup loads the configured model into the manager."""
    model_path = (tmp_path / "models" / "warmup.pt").resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("model", encoding="utf-8")

    settings = RuntimeSettings(
        startup_validate_paths=True,
        warmup_enabled=True,
        warmup_model_source=str(model_path),
        warmup_skip_if_missing=False,
        models_dir=model_path.parent,
        outputs_dir=(tmp_path / "outputs").resolve(),
    )
    fake_manager = FakeModelManager()

    result = maybe_warmup_model(settings, model_manager=fake_manager)  # type: ignore[arg-type]

    assert result is True
    assert fake_manager.requested_sources
    assert fake_manager.requested_sources[0] == model_path


def test_load_runtime_settings_reads_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure runtime settings are driven by environment variables."""
    models_dir = (tmp_path / "models").resolve()
    outputs_dir = (tmp_path / "outputs").resolve()

    monkeypatch.setenv("YOLO_STARTUP_VALIDATE_PATHS", "false")
    monkeypatch.setenv("YOLO_WARMUP_ENABLED", "true")
    monkeypatch.setenv("YOLO_WARMUP_MODEL", "weights/custom.pt")
    monkeypatch.setenv("YOLO_WARMUP_SKIP_IF_MISSING", "false")
    monkeypatch.setenv("YOLO_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("YOLO_OUTPUTS_DIR", str(outputs_dir))

    settings = load_runtime_settings()

    assert settings.startup_validate_paths is False
    assert settings.warmup_enabled is True
    assert settings.warmup_model_source == "weights/custom.pt"
    assert settings.warmup_skip_if_missing is False
    assert settings.models_dir == models_dir
    assert settings.outputs_dir == outputs_dir
