"""Unit tests for the YOLO model manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.model_manager import ModelManager


class FakeYOLO:
    """Simple fake YOLO model used for cache tests."""

    init_calls: list[str] = []

    def __init__(self, model_source: str) -> None:
        """Record each fake model initialization."""
        self.model_source = model_source
        self.__class__.init_calls.append(model_source)


def test_model_manager_is_singleton() -> None:
    """Ensure the model manager returns the same singleton instance."""
    manager_a = ModelManager()
    manager_b = ModelManager()

    assert manager_a is manager_b


def test_model_manager_caches_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure repeated requests for the same model reuse the cached instance."""
    import src.core.model_manager as model_manager_module

    FakeYOLO.init_calls.clear()
    monkeypatch.setattr(model_manager_module, "YOLO", FakeYOLO)

    manager = ModelManager()
    manager.clear_cache()

    model_path = tmp_path / "cached-model.pt"
    model_path.write_text("placeholder", encoding="utf-8")

    model_a = manager.get_model(model_path)
    model_b = manager.get_model(model_path)

    assert model_a is model_b
    assert manager.cache_size() == 1
    assert len(FakeYOLO.init_calls) == 1
