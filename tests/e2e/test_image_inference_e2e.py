"""End-to-end tests for image inference using production wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.inference_service import InferenceService


pytestmark = pytest.mark.e2e


def test_image_inference_end_to_end_returns_standardized_predictions() -> None:
    """Ensure image inference runs against the sample assets and returns the standard schema."""
    service = InferenceService()

    predictions = service.predict_image(
        image_path=Path("outputs/sprint1_check_image.jpg"),
        model_source=Path("models/yolo11n.pt"),
    )

    assert isinstance(predictions, list)
    for prediction in predictions:
        assert set(prediction.keys()) == {"class", "confidence", "bbox"}
        assert isinstance(prediction["class"], str)
        assert isinstance(prediction["confidence"], float)
        assert isinstance(prediction["bbox"], list)
        assert len(prediction["bbox"]) == 4
