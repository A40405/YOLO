"""Unit tests for the realtime webcam inference service."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.webcam_inference_service import WebcamInferenceService


class FakeVideoCapture:
    """Simple fake webcam capture object."""

    def __init__(self, frames: list[object], opened: bool = True) -> None:
        """Store frames and open state."""
        self._frames = frames
        self._opened = opened
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        """Return whether the fake webcam is open."""
        return self._opened

    def read(self) -> tuple[bool, object | None]:
        """Return the next frame if available."""
        if self._index >= len(self._frames):
            return False, None

        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        """Record release calls."""
        self.released = True


class FakeDetector:
    """Simple fake detector for webcam tests."""

    def __init__(self, results: list[object]) -> None:
        """Store fake result objects to return."""
        self._results = results
        self.calls: list[tuple[Path, object]] = []

    def predict_frame(self, model_source: str | Path, frame: object, **_: object) -> list[object]:
        """Record calls and return fake results."""
        self.calls.append((Path(model_source), frame))
        return self._results


class FakeTensorScalar:
    """Simple fake tensor scalar for formatting."""

    def __init__(self, value: float) -> None:
        """Store scalar value."""
        self._value = value

    def item(self) -> float:
        """Return scalar value."""
        return self._value


class FakeTensorList:
    """Simple fake tensor list for formatting."""

    def __init__(self, values: list[float]) -> None:
        """Store list values."""
        self._values = values

    def tolist(self) -> list[float]:
        """Return list values."""
        return self._values


class FakeXYXY:
    """Simple fake XYXY wrapper for formatting."""

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
        self.conf = FakeTensorScalar(0.75)
        self.xyxy = FakeXYXY([1.0, 2.0, 3.0, 4.0])


class FakeResult:
    """Simple fake raw result container."""

    def __init__(self) -> None:
        """Create fake names and boxes."""
        self.names = {0: "person"}
        self.boxes = [FakeBox()]


def test_webcam_service_handles_camera_open_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure webcam open failures are handled gracefully."""
    import src.services.webcam_inference_service as webcam_module

    monkeypatch.setattr(webcam_module.cv2, "VideoCapture", lambda _: FakeVideoCapture(frames=[], opened=False))
    service = WebcamInferenceService(detector=FakeDetector(results=[]))

    with pytest.raises(ValueError, match="Unable to open webcam"):
        service.run(model_source="model.pt", camera_index=0)


def test_webcam_service_quits_on_q(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure webcam processing stops when q is pressed."""
    import src.services.webcam_inference_service as webcam_module

    fake_capture = FakeVideoCapture(frames=["frame-1"], opened=True)
    detector = FakeDetector(results=[FakeResult()])
    shown_frames: list[object] = []

    monkeypatch.setattr(webcam_module.cv2, "VideoCapture", lambda _: fake_capture)
    monkeypatch.setattr(webcam_module.cv2, "imshow", lambda *_: shown_frames.append("shown"))
    monkeypatch.setattr(webcam_module.cv2, "waitKey", lambda _: ord("q"))
    monkeypatch.setattr(webcam_module.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(webcam_module, "annotate_frame", lambda frame, _: f"annotated-{frame}")
    monkeypatch.setattr(WebcamInferenceService, "_overlay_fps", staticmethod(lambda frame, _: frame))

    service = WebcamInferenceService(detector=detector)
    model_path = tmp_path / "model.pt"
    model_path.write_text("model", encoding="utf-8")

    summary = service.run(model_source=model_path, camera_index=0)

    assert summary == {
        "success": True,
        "message": "Webcam inference stopped by user",
    }
    assert detector.calls == [(model_path, "frame-1")]
    assert shown_frames == ["shown"]
