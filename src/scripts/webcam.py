"""CLI entry point for Sprint 5 realtime webcam inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.webcam_inference_service import WebcamInferenceService


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for webcam inference."""
    parser = argparse.ArgumentParser(description="Run realtime webcam inference with YOLO.")
    parser.add_argument(
        "--model",
        dest="model_path",
        required=True,
        help="Path to the YOLO model file.",
    )
    parser.add_argument(
        "--camera-index",
        dest="camera_index",
        type=int,
        default=0,
        help="Camera index to open.",
    )
    return parser


def main() -> int:
    """Open the webcam, run realtime inference, and display annotated frames."""
    args = build_parser().parse_args()
    service = WebcamInferenceService()
    try:
        summary = service.run(model_source=Path(args.model_path), camera_index=args.camera_index)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        logger.exception("Webcam inference failed: {}", exc)
        print(json.dumps({"success": False, "message": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
