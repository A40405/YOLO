"""API tests for the FastAPI serving layer."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from src.api.app import create_app
from src.api.routes import get_inference_service


pytestmark = pytest.mark.anyio


class FakeInferenceService:
    """Simple fake inference service used for API tests."""

    def __init__(self, *, should_fail: Exception | None = None) -> None:
        """Configure optional failure behavior."""
        self._should_fail = should_fail
        self.calls: list[tuple[str, str]] = []

    def predict_image(self, image_path: str, model_source: str) -> list[dict[str, object]]:
        """Record the request and return fake standardized predictions."""
        self.calls.append((str(image_path), str(model_source)))
        if self._should_fail is not None:
            raise self._should_fail
        return [
            {
                "class": "person",
                "confidence": 0.91,
                "bbox": [1.0, 2.0, 3.0, 4.0],
            }
        ]


async def test_health_endpoint_returns_ok() -> None:
    """Ensure the health endpoint returns a structured success response."""
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "ok",
    }


async def test_predict_endpoint_returns_predictions(sample_image_path: Path, yolo_model_source: str) -> None:
    """Ensure the predict endpoint returns validated prediction output."""
    app = create_app()
    fake_service = FakeInferenceService()
    image_path = str(sample_image_path)
    app.dependency_overrides[get_inference_service] = lambda: fake_service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/predict",
            json={
                "image_path": image_path,
                "model_source": yolo_model_source,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "image_path": image_path,
        "model_source": yolo_model_source,
        "predictions": [
            {
                "class": "person",
                "confidence": 0.91,
                "bbox": [1.0, 2.0, 3.0, 4.0],
            }
        ],
    }
    assert fake_service.calls == [(image_path, yolo_model_source)]


async def test_predict_endpoint_returns_structured_file_not_found_error(yolo_model_source: str) -> None:
    """Ensure missing file errors are converted into structured 404 responses."""
    app = create_app()
    app.dependency_overrides[get_inference_service] = lambda: FakeInferenceService(
        should_fail=FileNotFoundError("Image not found: /missing.jpg")
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/predict",
            json={
                "image_path": "/missing.jpg",
                "model_source": yolo_model_source,
            },
        )

    assert response.status_code == 404
    assert response.json() == {
        "success": False,
        "message": "Image not found: /missing.jpg",
    }


async def test_predict_endpoint_rejects_invalid_request_payload() -> None:
    """Ensure request validation rejects malformed payloads."""
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/predict",
            json={
                "image_path": "",
            },
        )

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"]


async def test_openapi_schema_includes_required_routes() -> None:
    """Ensure OpenAPI remains valid and exposes the required routes."""
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/health" in paths
    assert "/api/v1/predict" in paths
