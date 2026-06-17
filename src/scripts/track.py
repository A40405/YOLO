"""CLI entry point for Sprint 10 offline video tracking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.tracking_service import TrackingService


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for offline tracking."""
    parser = argparse.ArgumentParser(description="Run offline object tracking with YOLO detections.")
    parser.add_argument("video_path", help="Path to the input video.")
    parser.add_argument(
        "--model",
        dest="model_path",
        required=True,
        help="Path to the YOLO model file.",
    )
    return parser


def main() -> int:
    """Parse CLI arguments and run tracking on a video file."""
    args = build_parser().parse_args()
    service = TrackingService()

    try:
        summary = service.track_video(
            video_path=Path(args.video_path),
            model_source=Path(args.model_path),
            outputs_dir=PROJECT_ROOT / "outputs",
        )
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        logger.exception("Video tracking failed: {}", exc)
        print(json.dumps({"success": False, "message": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
