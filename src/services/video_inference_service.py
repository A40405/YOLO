"""Offline video inference service."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import cv2
from loguru import logger

from src.core.detector import Detector
from src.services.inference_service import standardize_predictions
from src.utils.video_utils import (
    annotate_frame,
    build_output_video_path,
    create_video_writer,
    read_video_metadata,
    validate_video_path,
)


class VideoInferenceSummary(TypedDict):
    """Summary output for offline video inference."""

    success: bool
    input_video: str
    output_video: str
    frames_processed: int
    fps: float
    total_predictions: int


class VideoInferenceService:
    """Service layer orchestration for offline video inference."""

    def __init__(self, detector: Detector | None = None) -> None:
        """Initialize the service with a detector dependency."""
        self._detector = detector or Detector()

    def predict_video(
        self,
        video_path: str | Path,
        model_source: str | Path,
        outputs_dir: str | Path,
    ) -> VideoInferenceSummary:
        """Generate an annotated output video for an input video path."""
        input_video_path = validate_video_path(video_path)
        output_video_path = build_output_video_path(input_video_path, outputs_dir)

        capture = cv2.VideoCapture(str(input_video_path))
        if not capture.isOpened():
            message = f"Unable to open video: {input_video_path}"
            logger.error(message)
            raise ValueError(message)

        try:
            metadata = read_video_metadata(capture)
            writer = create_video_writer(output_video_path, metadata)
            if not writer.isOpened():
                message = f"Unable to create output video: {output_video_path}"
                logger.error(message)
                raise ValueError(message)

            try:
                frame_index = 0
                total_predictions = 0

                while True:
                    success, frame = capture.read()
                    if not success:
                        break

                    raw_results = self._detector.predict_frame(model_source=model_source, frame=frame, verbose=False)
                    predictions = standardize_predictions(raw_results)
                    total_predictions += len(predictions)
                    annotated_frame = annotate_frame(frame, predictions)
                    writer.write(annotated_frame)
                    frame_index += 1
            finally:
                writer.release()

            logger.info("Annotated {} frames into {}", frame_index, output_video_path)
            return {
                "success": True,
                "input_video": str(input_video_path),
                "output_video": str(output_video_path),
                "frames_processed": frame_index,
                "fps": metadata["fps"],
                "total_predictions": total_predictions,
            }
        finally:
            capture.release()
