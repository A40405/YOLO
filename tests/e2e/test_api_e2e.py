"""End-to-end tests for the production FastAPI application."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from src.api.app import create_app


pytestmark = [pytest.mark.e2e, pytest.mark.anyio]


async def test_api_end_to_end_health_and_predict(sample_image_path: Path, yolo_model_source: str) -> None:
    """Ensure the production API wiring serves health and predict successfully."""
    image_path = str(sample_image_path)
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        health_response = await client.get("/api/v1/health")
        predict_response = await client.post(
            "/api/v1/predict",
            json={
                "image_path": image_path,
                "model_source": yolo_model_source,
            },
        )

    assert health_response.status_code == 200
    assert health_response.json() == {
        "success": True,
        "message": "ok",
    }

    assert predict_response.status_code == 200
    payload = predict_response.json()
    assert payload["success"] is True
    assert payload["image_path"] == image_path
    assert payload["model_source"] == yolo_model_source
    assert isinstance(payload["predictions"], list)
    for prediction in payload["predictions"]:
        assert set(prediction.keys()) == {"class", "confidence", "bbox"}
