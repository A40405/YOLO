"""Image inference service for standardized YOLO predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from loguru import logger
from PIL import Image, UnidentifiedImageError

from src.core.detector import Detector


Prediction = TypedDict(
    "Prediction",
    {
        "class": str,
        "confidence": float,
        "bbox": list[float],
    },
)


class InferenceService:
    """Service layer for image-only YOLO inference."""

    def __init__(self, detector: Detector | None = None) -> None:
        """Initialize the service with a detector dependency."""
        self._detector = detector or Detector()

    def predict_image(self, image_path: str | Path, model_source: str | Path) -> list[dict[str, Any]]:
        """Run inference on an image path and return standardized predictions."""
        resolved_image_path = Path(image_path)
        if not resolved_image_path.exists():
            message = f"Image not found: {resolved_image_path}"
            logger.error(message)
            raise FileNotFoundError(message)

        if not resolved_image_path.is_file():
            message = f"Image path is not a file: {resolved_image_path}"
            logger.error(message)
            raise ValueError(message)

        try:
            with Image.open(resolved_image_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            message = f"Invalid image: {resolved_image_path}"
            logger.exception("{}: {}", message, exc)
            raise ValueError(message) from exc

        try:
            results = self._detector.predict_image(model_source=model_source, image_path=resolved_image_path, verbose=False)
            predictions = standardize_predictions(results)
            logger.info("Generated {} predictions for {}", len(predictions), resolved_image_path)
            return predictions
        except Exception as exc:
            logger.exception("Image inference failed for {}: {}", resolved_image_path, exc)
            raise

def standardize_predictions(results: list[Any]) -> list[Prediction]:
    """Convert raw Ultralytics results to the standardized output schema."""
    predictions: list[Prediction] = []

    for result in results:
        names: dict[int, str] = result.names
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = round(float(box.conf.item()), 4)
            bbox = [round(float(value), 2) for value in box.xyxy[0].tolist()]
            predictions.append(
                {
                    "class": names[class_id],
                    "confidence": confidence,
                    "bbox": bbox,
                }
            )

    return predictions
