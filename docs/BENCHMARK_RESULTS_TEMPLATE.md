# Benchmark Results Template

## Metadata

- Date:
- Environment:
- Python version:
- `uv` version:
- GPU:
- Model path:

## Inference Benchmark

```json
{
  "benchmark_type": "inference",
  "model_path": "models/yolo11n.pt",
  "image_path": "outputs/sprint1_check_image.jpg",
  "video_path": "outputs/sprint4_sample.mp4",
  "metrics": {
    "model_load_time_ms": 0.0,
    "image_latency_ms": {
      "runs": 3,
      "average": 0.0,
      "min": 0.0,
      "max": 0.0,
      "prediction_counts": [0, 0, 0]
    },
    "video_fps": {
      "status": "ok",
      "runs": 1,
      "average": 0.0,
      "min": 0.0,
      "max": 0.0,
      "frame_counts": [0]
    },
    "webcam_fps": {
      "status": "skipped",
      "message": "Webcam not available."
    }
  }
}
```

## Tracking Benchmark

```json
{
  "benchmark_type": "tracking",
  "model_path": "models/yolo11n.pt",
  "video_path": "outputs/sprint4_sample.mp4",
  "metrics": {
    "tracking_fps": {
      "runs": 1,
      "average": 0.0,
      "min": 0.0,
      "max": 0.0,
      "frame_counts": [0]
    },
    "people_counting_throughput": {
      "runs": 1,
      "average": 0.0,
      "min": 0.0,
      "max": 0.0,
      "people_counts": [0]
    }
  }
}
```

## Notes

- Record whether webcam metrics were measured or skipped.
- Record whether benchmark runs used CPU or GPU.
- Preserve raw JSON output for comparison across releases.
