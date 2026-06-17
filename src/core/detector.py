"""Core detector wrapper around cached YOLO model execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from numpy import ndarray

from src.core.model_manager import ModelManager


class Detector:
    """Run YOLO predictions through the central model manager."""

    def __init__(self, model_manager: ModelManager | None = None) -> None:
        """Initialize the detector with a singleton model manager by default."""
        self._model_manager = model_manager or ModelManager()

    def predict_image(self, model_source: str | Path, image_path: str | Path, **kwargs: Any) -> list[Any]:
        """Run YOLO prediction for an image path using a centrally managed cached model."""
        return self._predict(model_source=model_source, source=str(image_path), **kwargs)

    def predict_frame(self, model_source: str | Path, frame: ndarray, **kwargs: Any) -> list[Any]:
        """Run YOLO prediction for an in-memory frame using a centrally managed cached model."""
        return self._predict(model_source=model_source, source=frame, **kwargs)

    def _predict(self, model_source: str | Path, source: Any, **kwargs: Any) -> list[Any]:
        """Run YOLO prediction using a centrally managed cached model."""
        try:
            model = self._model_manager.get_model(model_source)
            logger.info("Running detection with model {}", model_source)
            return model.predict(source=source, **kwargs)
        except Exception as exc:
            logger.exception("Detection failed for model {} and source {}: {}", model_source, source, exc)
            raise
