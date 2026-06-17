"""Validate YOLO model download and sample inference for Sprint 1."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_IMAGE_PATH = PROJECT_ROOT / "outputs" / "sprint1_check_image.jpg"
MODEL_NAME = "yolo11n.pt"
MODEL_PATH = PROJECT_ROOT / "models" / MODEL_NAME


def create_sample_image(image_path: Path) -> Path:
    """Create a simple synthetic image for inference validation."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 640), color=(240, 240, 240))
    image.save(image_path)
    return image_path


def collect_yolo_status() -> dict[str, Any]:
    """Download the YOLO model if needed and run sample inference."""
    sample_image_path = create_sample_image(OUTPUT_IMAGE_PATH)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=str(sample_image_path), verbose=False)
    first_result = results[0]

    return {
        "success": True,
        "model_name": MODEL_NAME,
        "model_path": str(Path(model.ckpt_path).resolve()) if model.ckpt_path else str(MODEL_PATH),
        "sample_image": str(sample_image_path),
        "inference_completed": True,
        "result_count": len(results),
        "boxes_detected": len(first_result.boxes),
        "image_shape": list(first_result.orig_shape),
    }


def main() -> int:
    """Run the YOLO validation script."""
    try:
        status = collect_yolo_status()
        logger.info("YOLO validation completed using {}", status["model_name"])
        print(json.dumps(status, indent=2))
        return 0
    except Exception as exc:  # pragma: no cover - exercised by runtime validation
        logger.exception("YOLO validation failed: {}", exc)
        print(
            json.dumps(
                {
                    "success": False,
                    "message": str(exc),
                    "model_name": MODEL_NAME,
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
