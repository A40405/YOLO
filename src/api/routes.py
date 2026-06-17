"""FastAPI routes for image inference and health checks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.services.inference_service import InferenceService


router = APIRouter(prefix="/api/v1", tags=["api"])


class HealthResponse(BaseModel):
    """Response model for the health endpoint."""

    success: bool = True
    message: str = "ok"


class PredictRequest(BaseModel):
    """Request model for image inference."""

    image_path: str = Field(..., min_length=1, description="Path to the input image file.")
    model_source: str = Field(..., min_length=1, description="Path or identifier for the YOLO model.")


class PredictionResponseItem(BaseModel):
    """Response model for a single standardized prediction."""

    class_name: str = Field(..., alias="class")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: list[float] = Field(..., min_length=4, max_length=4)

    model_config = {
        "populate_by_name": True,
    }


class PredictResponse(BaseModel):
    """Response model for image inference results."""

    success: bool
    image_path: str
    model_source: str
    predictions: list[PredictionResponseItem]


def get_inference_service() -> InferenceService:
    """Provide the image inference service for dependency injection."""
    return InferenceService()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return a simple API health status."""
    return HealthResponse()


@router.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> PredictResponse:
    """Run image inference through the service layer and return standardized predictions."""
    predictions = service.predict_image(
        image_path=Path(request.image_path),
        model_source=Path(request.model_source),
    )
    return PredictResponse(
        success=True,
        image_path=request.image_path,
        model_source=request.model_source,
        predictions=[PredictionResponseItem.model_validate(prediction) for prediction in predictions],
    )
