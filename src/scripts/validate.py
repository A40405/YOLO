"""CLI entrypoint for configuration-driven YOLO validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.training_service import TrainingService


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for validation commands."""
    parser = argparse.ArgumentParser(description="Validate a YOLO model using YAML configuration files.")
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="Path to the dataset YAML configuration file.",
    )
    parser.add_argument(
        "--train-config",
        default="configs/train.yaml",
        help="Path to the training YAML configuration file.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Evaluation split to use: train, val, or test.",
    )
    return parser


def main() -> int:
    """Run YOLO validation and print a structured summary."""
    parser = build_parser()
    args = parser.parse_args()
    service = TrainingService()

    try:
        summary = service.validate(
            data_config_path=Path(args.data_config),
            train_config_path=Path(args.train_config),
            split=args.split,
        )
    except Exception as exc:
        logger.exception("Validation command failed: {}", exc)
        print(
            json.dumps(
                {
                    "success": False,
                    "message": str(exc),
                    "data_config": args.data_config,
                    "train_config": args.train_config,
                    "split": args.split,
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
