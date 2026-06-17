"""CLI entry point for Sprint 3 image inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.inference_service import InferenceService


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run image inference with YOLO.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument(
        "--model",
        dest="model_path",
        required=True,
        help="Path to the YOLO model file.",
    )
    return parser


def main() -> int:
    """Parse CLI arguments and run image inference."""
    args = build_parser().parse_args()
    service = InferenceService()

    try:
        predictions = service.predict_image(
            image_path=Path(args.image_path),
            model_source=Path(args.model_path),
        )
        print(json.dumps(predictions, indent=2))
        return 0
    except Exception as exc:
        logger.exception("Prediction script failed: {}", exc)
        print(
            json.dumps(
                {
                    "success": False,
                    "message": str(exc),
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
