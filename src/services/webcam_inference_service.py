"""Realtime webcam inference service."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import cv2
from loguru import logger

from src.core.detector import Detector
from src.services.inference_service import standardize_predictions
from src.utils.video_utils import annotate_frame


class WebcamInferenceSummary(TypedDict):
    """Summary output for webcam inference."""

    success: bool
    message: str


class WebcamInferenceService:
    """Service layer orchestration for realtime webcam inference."""

    def __init__(self, detector: Detector | None = None) -> None:
        """Initialize the service with a detector dependency."""
        self._detector = detector or Detector()

    def run(self, model_source: str | Path, camera_index: int = 0) -> WebcamInferenceSummary:
        """Open the webcam, run inference, and display annotated frames."""
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            message = f"Unable to open webcam at index {camera_index}"
            logger.error(message)
            raise ValueError(message)

        logger.info("Webcam opened at index {}", camera_index)
        previous_time = time.perf_counter()

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    message = "Failed to read frame from webcam"
                    logger.error(message)
                    raise ValueError(message)

                raw_results = self._detector.predict_frame(model_source=model_source, frame=frame, verbose=False)
                predictions = standardize_predictions(raw_results)
                annotated_frame = annotate_frame(frame, predictions)

                current_time = time.perf_counter()
                elapsed = current_time - previous_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                previous_time = current_time

                self._overlay_fps(annotated_frame, fps)
                cv2.imshow("YOLO Webcam Inference", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit key detected, stopping webcam inference")
                    break

            return {
                "success": True,
                "message": "Webcam inference stopped by user",
            }
        finally:
            capture.release()
            cv2.destroyAllWindows()

    @staticmethod
    def _overlay_fps(frame: object, fps: float) -> object:
        """Overlay the current FPS value onto a frame."""
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return frame
