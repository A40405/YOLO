"""Unit tests for the offline video inference service."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.video_inference_service import VideoInferenceService


class FakeVideoCapture:
    """Simple fake capture object for video inference tests."""

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
    """Simple fake writer object for video inference tests."""

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
    """Simple fake detector for video inference tests."""

    def __init__(self, results: list[object]) -> None:
        """Store fake result objects to return."""
        self._results = results
        self.calls: list[tuple[Path, object]] = []

    def predict_frame(self, model_source: str | Path, frame: object, **_: object) -> list[object]:
        """Record calls and return fake raw results."""
        self.calls.append((Path(model_source), frame))
        return self._results


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

    def __init__(self) -> None:
        """Create one fake detection box."""
        self.cls = FakeTensorScalar(0)
        self.conf = FakeTensorScalar(0.9)
        self.xyxy = FakeXYXY([1.0, 2.0, 3.0, 4.0])


class FakeResult:
    """Simple fake raw result container."""

    def __init__(self) -> None:
        """Create fake names and boxes."""
        self.names = {0: "person"}
        self.boxes = [FakeBox()]


def test_predict_video_rejects_missing_video(tmp_path: Path) -> None:
    """Ensure missing video paths are rejected before processing."""
    service = VideoInferenceService(detector=FakeDetector(results=[]))

    with pytest.raises(FileNotFoundError):
        service.predict_video(tmp_path / "missing.mp4", tmp_path / "model.pt", tmp_path)


def test_predict_video_processes_frames(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure the service processes frames and returns a summary."""
    import src.services.video_inference_service as video_module

    video_path = tmp_path / "input.mp4"
    model_path = tmp_path / "model.pt"
    video_path.write_text("video", encoding="utf-8")
    model_path.write_text("model", encoding="utf-8")

    fake_capture = FakeVideoCapture(frames=["frame-1", "frame-2"])
    fake_writer = FakeVideoWriter()
    detector = FakeDetector(results=[FakeResult()])

    monkeypatch.setattr(video_module.cv2, "VideoCapture", lambda _: fake_capture)
    monkeypatch.setattr(video_module, "read_video_metadata", lambda _: {"fps": 12.0, "width": 320, "height": 240, "frame_count": 2})
    monkeypatch.setattr(video_module, "create_video_writer", lambda *_: fake_writer)
    monkeypatch.setattr(video_module, "annotate_frame", lambda frame, _: f"annotated-{frame}")

    service = VideoInferenceService(detector=detector)
    summary = service.predict_video(video_path, model_path, tmp_path)

    assert summary == {
        "success": True,
        "input_video": str(video_path),
        "output_video": str(tmp_path / "input_annotated.mp4"),
        "frames_processed": 2,
        "fps": 12.0,
        "total_predictions": 2,
    }
    assert fake_writer.frames == ["annotated-frame-1", "annotated-frame-2"]
    assert detector.calls == [(model_path, "frame-1"), (model_path, "frame-2")]
