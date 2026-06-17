"""Unit tests for offline video helper utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.utils.video_utils import annotate_frame, build_output_video_path, validate_video_path


def test_validate_video_path_accepts_supported_extension(tmp_path: Path) -> None:
    """Ensure supported video paths are accepted."""
    video_path = tmp_path / "sample.mp4"
    video_path.write_text("video", encoding="utf-8")

    assert validate_video_path(video_path) == video_path


def test_validate_video_path_rejects_invalid_extension(tmp_path: Path) -> None:
    """Ensure unsupported video formats are rejected."""
    video_path = tmp_path / "sample.mkv"
    video_path.write_text("video", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported video format"):
        validate_video_path(video_path)


def test_build_output_video_path_stays_in_outputs_directory(tmp_path: Path) -> None:
    """Ensure output videos are always placed in the outputs directory."""
    output_path = build_output_video_path("input.mov", tmp_path)

    assert output_path == tmp_path / "input_annotated.mov"


def test_annotate_frame_draws_on_frame() -> None:
    """Ensure predictions produce visible annotations on the frame."""
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    predictions = [{"class": "person", "confidence": 0.91, "bbox": [2.0, 2.0, 20.0, 20.0]}]

    annotated_frame = annotate_frame(frame, predictions)

    assert not np.array_equal(frame, annotated_frame)
