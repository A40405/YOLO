"""Pure helper utilities for offline video processing."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
VIDEO_CODEC_BY_EXTENSION = {
    ".mp4": "mp4v",
    ".avi": "XVID",
    ".mov": "mp4v",
}


class VideoMetadata(TypedDict):
    """Basic video metadata used for offline processing."""

    fps: float
    width: int
    height: int
    frame_count: int


def validate_video_path(video_path: str | Path) -> Path:
    """Validate a local video path and its supported extension."""
    resolved_video_path = Path(video_path)
    if not resolved_video_path.exists():
        raise FileNotFoundError(f"Video not found: {resolved_video_path}")

    if not resolved_video_path.is_file():
        raise ValueError(f"Video path is not a file: {resolved_video_path}")

    if resolved_video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format: {resolved_video_path.suffix}")

    return resolved_video_path


def build_output_video_path(input_video_path: str | Path, outputs_dir: str | Path) -> Path:
    """Build the annotated output path inside the outputs directory."""
    input_path = Path(input_video_path)
    output_dir = Path(outputs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}_annotated{input_path.suffix.lower()}"


def read_video_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    """Read video properties with safe defaults for FPS and dimensions."""
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    return {
        "fps": fps if fps > 0 else 30.0,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def create_video_writer(output_path: str | Path, metadata: VideoMetadata) -> cv2.VideoWriter:
    """Create a video writer that preserves the input FPS when possible."""
    output_suffix = Path(output_path).suffix.lower()
    codec = VIDEO_CODEC_BY_EXTENSION.get(output_suffix, "mp4v")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(
        str(output_path),
        fourcc,
        metadata["fps"],
        (metadata["width"], metadata["height"]),
    )


def annotate_frame(frame: np.ndarray, predictions: list[dict[str, object]]) -> np.ndarray:
    """Draw standardized predictions on a video frame."""
    annotated_frame = frame.copy()

    for prediction in predictions:
        bbox = prediction["bbox"]
        label = f"{prediction['class']} {prediction['confidence']:.2f}"
        x1, y1, x2, y2 = [int(value) for value in bbox]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated_frame
