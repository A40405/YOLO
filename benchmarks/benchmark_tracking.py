"""Reusable benchmarking script for tracking workflows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.tracking_service import TrackingService


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for tracking benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark YOLO tracking workflows.")
    parser.add_argument("--model", dest="model_path", required=True, help="Path to the YOLO model file.")
    parser.add_argument("--video", dest="video_path", default="outputs/sprint4_sample.mp4", help="Path to the input video.")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated tracking benchmark runs.")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory for annotated tracking outputs.")
    return parser


def main() -> int:
    """Execute tracking benchmarks and print a JSON report."""
    args = build_parser().parse_args()
    report = benchmark_tracking(
        model_path=Path(args.model_path),
        video_path=Path(args.video_path),
        runs=args.runs,
        outputs_dir=Path(args.outputs_dir),
    )
    print(json.dumps(report, indent=2))
    return 0


def benchmark_tracking(
    *,
    model_path: Path,
    video_path: Path,
    runs: int,
    outputs_dir: Path,
) -> dict[str, Any]:
    """Run reusable tracking benchmarks and return a JSON-serializable report."""
    service = TrackingService()
    tracking_fps_values: list[float] = []
    throughput_values: list[float] = []
    people_counts: list[int] = []
    frame_counts: list[int] = []

    for run_index in range(runs):
        run_output_dir = outputs_dir / f"benchmark_tracking_run_{run_index + 1}"
        start_time = time.perf_counter()
        summary = service.track_video(
            video_path=video_path,
            model_source=model_path,
            outputs_dir=run_output_dir,
        )
        elapsed = time.perf_counter() - start_time
        tracking_fps_values.append(summary["frames_processed"] / elapsed if elapsed > 0 else 0.0)
        throughput_values.append(summary["people_count"] / elapsed if elapsed > 0 else 0.0)
        people_counts.append(summary["people_count"])
        frame_counts.append(summary["frames_processed"])

    return {
        "benchmark_type": "tracking",
        "model_path": str(model_path.resolve()),
        "video_path": str(video_path.resolve()),
        "metrics": {
            "tracking_fps": {
                "runs": runs,
                "average": round(sum(tracking_fps_values) / len(tracking_fps_values), 2),
                "min": round(min(tracking_fps_values), 2),
                "max": round(max(tracking_fps_values), 2),
                "frame_counts": frame_counts,
            },
            "people_counting_throughput": {
                "runs": runs,
                "average": round(sum(throughput_values) / len(throughput_values), 4),
                "min": round(min(throughput_values), 4),
                "max": round(max(throughput_values), 4),
                "people_counts": people_counts,
            },
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
