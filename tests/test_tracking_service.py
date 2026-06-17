"""Unit tests for offline video tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.services.tracking_service import TrackingService


class FakeVideoCapture:
    """Simple fake capture object for tracking tests."""

    def __init__(self, frames: list[object], opened: bool = True) -> None:
        """Store fake frames and open state."""
        self._frames = frames
        self._opened = opened
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        """Return whether the fake capture is open."""
        return self._opened

    def read(self) -> tuple[bool, object | None]:
        """Return the next fake frame if available."""
        if self._index >= len(self._frames):
            return False, None

        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        """Record release calls."""
        self.released = True


class FakeVideoWriter:
    """Simple fake writer object for tracking tests."""

    def __init__(self, opened: bool = True) -> None:
        """Store open state and written frames."""
        self._opened = opened
        self.frames: list[object] = []
        self.released = False

    def isOpened(self) -> bool:
        """Return whether the fake writer is open."""
        return self._opened

    def write(self, frame: object) -> None:
        """Record written frames."""
        self.frames.append(frame)

    def release(self) -> None:
        """Record release calls."""
        self.released = True


class FakeDetector:
    """Simple fake detector for tracking tests."""

    def __init__(self, results_by_call: list[list[object]]) -> None:
        """Store fake raw result objects per frame."""
        self._results_by_call = results_by_call
        self.calls: list[tuple[Path, object]] = []

    def predict_frame(self, model_source: str | Path, frame: object, **_: object) -> list[object]:
        """Record calls and return frame-specific raw results."""
        self.calls.append((Path(model_source), frame))
        return self._results_by_call[len(self.calls) - 1]


class FakeTensorScalar:
    """Simple fake tensor scalar for prediction formatting."""

    def __init__(self, value: float) -> None:
        """Store scalar value."""
        self._value = value

    def item(self) -> float:
        """Return scalar value."""
        return self._value


class FakeTensorList:
    """Simple fake tensor list for prediction formatting."""

    def __init__(self, values: list[float]) -> None:
        """Store list values."""
        self._values = values

    def tolist(self) -> list[float]:
        """Return list values."""
        return self._values


class FakeXYXY:
    """Simple fake XYXY wrapper for prediction formatting."""

    def __init__(self, values: list[float]) -> None:
        """Store bbox values."""
        self._values = values

    def __getitem__(self, _: int) -> FakeTensorList:
        """Return bbox values."""
        return FakeTensorList(self._values)


class FakeBox:
    """Simple fake detection box."""

    def __init__(self, class_id: int, confidence: float, bbox: list[float]) -> None:
        """Create one fake detection box."""
        self.cls = FakeTensorScalar(class_id)
        self.conf = FakeTensorScalar(confidence)
        self.xyxy = FakeXYXY(bbox)


class FakeResult:
    """Simple fake raw result container."""

    def __init__(self, boxes: list[FakeBox], names: dict[int, str]) -> None:
        """Create fake names and boxes."""
        self.names = names
        self.boxes = boxes


def test_update_tracks_preserves_track_id_and_counts_people() -> None:
    """Ensure one moving person keeps the same ID and increments unique people count once."""
    service = TrackingService(detector=FakeDetector(results_by_call=[]), iou_threshold=0.1)
    active_tracks = []
    next_track_id = 1

    frame_one_tracks, next_track_id = service._update_tracks(
        active_tracks=active_tracks,
        detections=[{"class": "person", "confidence": 0.95, "bbox": [10.0, 10.0, 30.0, 30.0]}],
        next_track_id=next_track_id,
    )
    frame_two_tracks, next_track_id = service._update_tracks(
        active_tracks=active_tracks,
        detections=[{"class": "person", "confidence": 0.94, "bbox": [12.0, 12.0, 32.0, 32.0]}],
        next_track_id=next_track_id,
    )

    assert frame_one_tracks[0]["track_id"] == 1
    assert frame_two_tracks[0]["track_id"] == 1
    assert frame_two_tracks[0]["history"] == [[20.0, 20.0], [22.0, 22.0]]
    assert next_track_id == 2


def test_update_tracks_assigns_new_ids_for_distinct_people() -> None:
    """Ensure distinct people receive distinct persistent IDs."""
    service = TrackingService(detector=FakeDetector(results_by_call=[]), iou_threshold=0.1)
    active_tracks = []

    frame_tracks, next_track_id = service._update_tracks(
        active_tracks=active_tracks,
        detections=[
            {"class": "person", "confidence": 0.95, "bbox": [10.0, 10.0, 30.0, 30.0]},
            {"class": "person", "confidence": 0.90, "bbox": [100.0, 100.0, 130.0, 130.0]},
        ],
        next_track_id=1,
    )

    assert [track["track_id"] for track in frame_tracks] == [1, 2]
    assert next_track_id == 3


def test_track_video_processes_frames_and_returns_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure offline tracking processes frames and returns a structured summary."""
    import src.services.tracking_service as tracking_module

    video_path = tmp_path / "input.mp4"
    model_path = tmp_path / "model.pt"
    video_path.write_text("video", encoding="utf-8")
    model_path.write_text("model", encoding="utf-8")

    fake_capture = FakeVideoCapture(
        frames=[
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
        ]
    )
    fake_writer = FakeVideoWriter()
    detector = FakeDetector(
        results_by_call=[
            [FakeResult(boxes=[FakeBox(0, 0.91, [1.0, 1.0, 5.0, 5.0])], names={0: "person"})],
            [FakeResult(boxes=[FakeBox(0, 0.92, [1.5, 1.5, 5.5, 5.5])], names={0: "person"})],
        ]
    )

    monkeypatch.setattr(tracking_module.cv2, "VideoCapture", lambda _: fake_capture)
    monkeypatch.setattr(tracking_module, "read_video_metadata", lambda _: {"fps": 12.0, "width": 16, "height": 16, "frame_count": 2})
    monkeypatch.setattr(tracking_module, "create_video_writer", lambda *_: fake_writer)

    service = TrackingService(detector=detector, iou_threshold=0.1)
    summary = service.track_video(video_path, model_path, tmp_path)

    assert summary["success"] is True
    assert summary["input_video"] == str(video_path)
    assert summary["output_video"] == str(tmp_path / "input_annotated.mp4")
    assert summary["frames_processed"] == 2
    assert summary["fps"] == 12.0
    assert summary["people_count"] == 1
    assert summary["total_tracks"] == 1
    assert len(summary["frame_results"]) == 2
    assert summary["frame_results"][0]["tracks"][0]["track_id"] == 1
    assert summary["frame_results"][1]["tracks"][0]["track_id"] == 1
    assert len(fake_writer.frames) == 2
    assert detector.calls[0][0] == model_path


def test_track_video_rejects_missing_video(tmp_path: Path) -> None:
    """Ensure missing video paths are rejected before processing."""
    service = TrackingService(detector=FakeDetector(results_by_call=[]))

    with pytest.raises(FileNotFoundError):
        service.track_video(tmp_path / "missing.mp4", tmp_path / "model.pt", tmp_path)
