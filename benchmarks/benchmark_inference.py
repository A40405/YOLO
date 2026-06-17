"""Reusable benchmarking script for inference workflows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.detector import Detector
from src.core.model_manager import ModelManager
from src.services.inference_service import InferenceService, standardize_predictions


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for inference benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark YOLO inference workflows.")
    parser.add_argument("--model", dest="model_path", required=True, help="Path to the YOLO model file.")
    parser.add_argument("--image", dest="image_path", default="outputs/sprint1_check_image.jpg", help="Path to the sample image.")
    parser.add_argument("--video", dest="video_path", default="outputs/sprint4_sample.mp4", help="Path to the sample video.")
    parser.add_argument("--image-runs", type=int, default=3, help="Number of repeated image inference runs.")
    parser.add_argument("--video-runs", type=int, default=1, help="Number of repeated video benchmarking runs.")
    parser.add_argument("--webcam-frames", type=int, default=30, help="Number of webcam frames to measure when available.")
    return parser


def main() -> int:
    """Execute inference benchmarks and print a JSON report."""
    args = build_parser().parse_args()
    report = benchmark_inference(
        model_path=Path(args.model_path),
        image_path=Path(args.image_path),
        video_path=Path(args.video_path),
        image_runs=args.image_runs,
        video_runs=args.video_runs,
        webcam_frames=args.webcam_frames,
    )
    print(json.dumps(report, indent=2))
    return 0


def benchmark_inference(
    *,
    model_path: Path,
    image_path: Path,
    video_path: Path,
    image_runs: int,
    video_runs: int,
    webcam_frames: int,
) -> dict[str, Any]:
    """Run reusable inference benchmarks and return a JSON-serializable report."""
    manager = ModelManager()
    detector = Detector(model_manager=manager)
    image_service = InferenceService(detector=detector)

    manager.clear_cache()
    model_load_start = time.perf_counter()
    manager.get_model(model_path)
    model_load_time_ms = round((time.perf_counter() - model_load_start) * 1000.0, 2)

    image_times_ms: list[float] = []
    image_prediction_counts: list[int] = []
    for _ in range(image_runs):
        start_time = time.perf_counter()
        predictions = image_service.predict_image(image_path=image_path, model_source=model_path)
        image_times_ms.append((time.perf_counter() - start_time) * 1000.0)
        image_prediction_counts.append(len(predictions))

    video_metrics = _benchmark_video(detector=detector, model_path=model_path, video_path=video_path, video_runs=video_runs)
    webcam_metrics = _benchmark_webcam(detector=detector, model_path=model_path, webcam_frames=webcam_frames)

    return {
        "benchmark_type": "inference",
        "model_path": str(model_path.resolve()),
        "image_path": str(image_path.resolve()),
        "video_path": str(video_path.resolve()),
        "metrics": {
            "model_load_time_ms": model_load_time_ms,
            "image_latency_ms": {
                "runs": image_runs,
                "average": round(sum(image_times_ms) / len(image_times_ms), 2),
                "min": round(min(image_times_ms), 2),
                "max": round(max(image_times_ms), 2),
                "prediction_counts": image_prediction_counts,
            },
            "video_fps": video_metrics,
            "webcam_fps": webcam_metrics,
        },
    }


def _benchmark_video(*, detector: Detector, model_path: Path, video_path: Path, video_runs: int) -> dict[str, Any]:
    """Measure frame throughput for offline video inference."""
    if not video_path.exists():
        return {"status": "skipped", "message": f"Video not found: {video_path}"}

    fps_results: list[float] = []
    frame_counts: list[int] = []

    for _ in range(video_runs):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return {"status": "error", "message": f"Unable to open video: {video_path}"}

        processed_frames = 0
        start_time = time.perf_counter()
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                raw_results = detector.predict_frame(model_source=model_path, frame=frame, verbose=False)
                standardize_predictions(raw_results)
                processed_frames += 1
        finally:
            capture.release()

        elapsed = time.perf_counter() - start_time
        fps_results.append(processed_frames / elapsed if elapsed > 0 else 0.0)
        frame_counts.append(processed_frames)

    return {
        "status": "ok",
        "runs": video_runs,
        "average": round(sum(fps_results) / len(fps_results), 2),
        "min": round(min(fps_results), 2),
        "max": round(max(fps_results), 2),
        "frame_counts": frame_counts,
    }


def _benchmark_webcam(*, detector: Detector, model_path: Path, webcam_frames: int) -> dict[str, Any]:
    """Measure live webcam throughput when a camera is available."""
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        return {"status": "skipped", "message": "Webcam not available."}

    processed_frames = 0
    start_time = time.perf_counter()
    try:
        while processed_frames < webcam_frames:
            success, frame = capture.read()
            if not success:
                break

            raw_results = detector.predict_frame(model_source=model_path, frame=frame, verbose=False)
            standardize_predictions(raw_results)
            processed_frames += 1
    finally:
        capture.release()

    elapsed = time.perf_counter() - start_time
    fps = processed_frames / elapsed if elapsed > 0 else 0.0
    return {
        "status": "ok" if processed_frames > 0 else "skipped",
        "frames": processed_frames,
        "fps": round(fps, 2),
    }


if __name__ == "__main__":
    raise SystemExit(main())
