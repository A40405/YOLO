"""Integration tests for offline tracking orchestration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.services.tracking_service import TrackingService


pytestmark = pytest.mark.integration


class FakeDetector:
    """Detector fake that returns frame-specific raw YOLO-like results."""

    def __init__(self, results_by_call: list[list[object]]) -> None:
        """Store results returned for each processed frame."""
        self._results_by_call = results_by_call
        self.calls: list[tuple[Path, np.ndarray]] = []

    def predict_frame(self, model_source: str | Path, frame: np.ndarray, **_: object) -> list[object]:
        """Record the frame call and return the configured fake results."""
        self.calls.append((Path(model_source), frame))
        return self._results_by_call[len(self.calls) - 1]


class FakeTensorScalar:
    """Simple scalar wrapper that mimics tensor item access."""

    def __init__(self, value: float) -> None:
        """Store the scalar value."""
        self._value = value

    def item(self) -> float:
        """Return the wrapped scalar value."""
        return self._value


class FakeTensorList:
    """Simple list wrapper mimicking tensor tolist access."""

    def __init__(self, values: list[float]) -> None:
        """Store list values."""
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
    """Simple prediction box object for tracking integration tests."""

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


def _create_sample_video(video_path: Path) -> None:
    """Create a tiny MP4 video for integration testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 12.0, (32, 32))
    if not writer.isOpened():
        raise RuntimeError("Failed to create integration test video.")

    try:
        for intensity in (20, 40):
            frame = np.full((32, 32, 3), intensity, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_tracking_service_processes_video_and_preserves_id(tmp_path: Path) -> None:
    """Ensure the tracking service preserves one person ID across frames using real video I/O."""
    video_path = tmp_path / "tracking.mp4"
    model_path = tmp_path / "model.pt"
    outputs_dir = tmp_path / "outputs"
    model_path.write_text("model", encoding="utf-8")
    _create_sample_video(video_path)

    detector = FakeDetector(
        results_by_call=[
            [FakeResult(boxes=[FakeBox(0, 0.91, [2.0, 2.0, 12.0, 12.0])], names={0: "person"})],
            [FakeResult(boxes=[FakeBox(0, 0.92, [3.0, 3.0, 13.0, 13.0])], names={0: "person"})],
        ]
    )

    summary = TrackingService(detector=detector, iou_threshold=0.1).track_video(
        video_path=video_path,
        model_source=model_path,
        outputs_dir=outputs_dir,
    )

    assert summary["success"] is True
    assert summary["frames_processed"] == 2
    assert summary["people_count"] == 1
    assert summary["total_tracks"] == 1
    assert summary["frame_results"][0]["tracks"][0]["track_id"] == 1
    assert summary["frame_results"][1]["tracks"][0]["track_id"] == 1
    assert Path(summary["output_video"]).exists()
