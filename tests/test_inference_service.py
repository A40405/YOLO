"""Unit tests for the image inference service."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.services.inference_service import InferenceService


class FakeTensorScalar:
    """Simple scalar wrapper mimicking tensor item access."""

    def __init__(self, value: float) -> None:
        """Store the scalar value."""
        self._value = value

    def item(self) -> float:
        """Return the wrapped scalar value."""
        return self._value


class FakeTensorList:
    """Simple list wrapper mimicking tensor tolist access."""

    def __init__(self, values: list[float]) -> None:
        """Store the list values."""
        self._values = values

    def tolist(self) -> list[float]:
        """Return the wrapped list values."""
        return self._values


class FakeXYXY:
    """Simple xyxy wrapper mimicking tensor indexing."""

    def __init__(self, values: list[float]) -> None:
        """Store one bounding box list."""
        self._values = values

    def __getitem__(self, _: int) -> FakeTensorList:
        """Return the stored box list regardless of index."""
        return FakeTensorList(self._values)


class FakeBox:
    """Simple prediction box object for schema tests."""

    def __init__(self, class_id: int, confidence: float, bbox: list[float]) -> None:
        """Store class, confidence, and bounding box values."""
        self.cls = FakeTensorScalar(class_id)
        self.conf = FakeTensorScalar(confidence)
        self.xyxy = FakeXYXY(bbox)


class FakeResult:
    """Simple raw result container for schema transformation tests."""

    def __init__(self, boxes: list[FakeBox], names: dict[int, str]) -> None:
        """Store fake boxes and class names."""
        self.boxes = boxes
        self.names = names


class FakeDetector:
    """Simple detector fake used by the service tests."""

    def __init__(self, results: list[FakeResult]) -> None:
        """Store raw results to return."""
        self._results = results
        self.calls: list[tuple[Path, Path]] = []

    def predict_image(self, model_source: str | Path, image_path: str | Path, **_: object) -> list[FakeResult]:
        """Record detector calls and return the configured fake results."""
        self.calls.append((Path(model_source), Path(image_path)))
        return self._results


def test_predict_image_returns_standardized_schema(tmp_path: Path) -> None:
    """Ensure the service converts raw results to the standardized schema."""
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), color=(255, 255, 255)).save(image_path)
    model_path = tmp_path / "model.pt"
    model_path.write_text("model", encoding="utf-8")

    fake_results = [
        FakeResult(
            boxes=[
                FakeBox(0, 0.91234, [1.0, 2.0, 101.0, 202.0]),
                FakeBox(1, 0.5, [10.5, 20.25, 30.75, 40.0]),
            ],
            names={0: "person", 1: "car"},
        )
    ]
    detector = FakeDetector(results=fake_results)
    service = InferenceService(detector=detector)

    predictions = service.predict_image(image_path=image_path, model_source=model_path)

    assert predictions == [
        {"class": "person", "confidence": 0.9123, "bbox": [1.0, 2.0, 101.0, 202.0]},
        {"class": "car", "confidence": 0.5, "bbox": [10.5, 20.25, 30.75, 40.0]},
    ]
    assert detector.calls == [(model_path, image_path)]


def test_predict_image_raises_for_missing_image(tmp_path: Path) -> None:
    """Ensure missing image paths are rejected before detector execution."""
    detector = FakeDetector(results=[])
    service = InferenceService(detector=detector)
    model_path = tmp_path / "model.pt"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        service.predict_image(image_path=tmp_path / "missing.jpg", model_source=model_path)


def test_predict_image_raises_for_invalid_image(tmp_path: Path) -> None:
    """Ensure invalid image files are rejected before detector execution."""
    detector = FakeDetector(results=[])
    service = InferenceService(detector=detector)
    image_path = tmp_path / "invalid.jpg"
    model_path = tmp_path / "model.pt"

    image_path.write_text("not an image", encoding="utf-8")
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid image"):
        service.predict_image(image_path=image_path, model_source=model_path)
