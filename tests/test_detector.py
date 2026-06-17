"""Unit tests for the core detector."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.detector import Detector
from src.core.model_manager import ModelManager


class FakeYOLO:
    """Simple fake YOLO model used for detector tests."""

    init_calls: list[str] = []

    def __init__(self, model_source: str) -> None:
        """Record model initialization for duplicate-load assertions."""
        self.model_source = model_source
        self.predict_calls: list[str] = []
        self.__class__.init_calls.append(model_source)

    def predict(self, source: str, **_: object) -> list[dict[str, str]]:
        """Record prediction calls and return a minimal fake result."""
        self.predict_calls.append(source)
        return [{"source": source, "model": self.model_source}]


def test_detector_reuses_cached_model_for_image_prediction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure the detector does not trigger duplicate YOLO loads for image paths."""
    import src.core.model_manager as model_manager_module

    FakeYOLO.init_calls.clear()
    monkeypatch.setattr(model_manager_module, "YOLO", FakeYOLO)

    manager = ModelManager()
    manager.clear_cache()
    detector = Detector(model_manager=manager)

    model_path = tmp_path / "detector-model.pt"
    image_path = tmp_path / "sample.jpg"
    model_path.write_text("placeholder", encoding="utf-8")
    image_path.write_text("image", encoding="utf-8")

    first_result = detector.predict_image(model_path, image_path)
    second_result = detector.predict_image(model_path, image_path)

    assert first_result == second_result
    assert manager.cache_size() == 1
    assert len(FakeYOLO.init_calls) == 1
