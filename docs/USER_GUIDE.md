# User Guide

## Purpose

This guide explains how to use the implemented `1.0.0` platform features from the repository root without reading source code.

Covered workflows:

- image inference
- video inference
- webcam inference
- training
- validation
- benchmarking
- API usage
- Docker usage

## Environment

Supported environment:

- WSL2 Ubuntu
- Python `3.11`
- `uv`

Work from the repository root:

```bash
uv sync --python 3.11
```

## Required Files and Directories

Before running commands, confirm these paths exist:

- `models/`
- `outputs/`
- `configs/data.yaml`
- `configs/train.yaml`

The default local model path used across the repository is:

```text
models/yolo11n.pt
```

## Image Inference

Command:

```bash
uv run --python 3.11 python src/scripts/predict.py outputs/sprint1_check_image.jpg --model models/yolo11n.pt
```

What it does:

- validates the image file
- loads the YOLO model through the core layer
- returns standardized JSON predictions

Output schema:

```json
[
  {
    "class": "person",
    "confidence": 0.91,
    "bbox": [0.0, 0.0, 100.0, 100.0]
  }
]
```

If the image is missing or invalid, the command returns JSON with:

- `success: false`
- `message`

## Offline Video Inference

Command:

```bash
uv run --python 3.11 python src/scripts/predict_video.py outputs/sprint4_sample.mp4 --model models/yolo11n.pt
```

What it does:

- validates the video path
- reads video metadata
- runs frame-by-frame inference
- writes an annotated video into `outputs/`

Output file pattern:

```text
outputs/<input-name>_annotated.<ext>
```

Returned summary fields include:

- `success`
- `input_video`
- `output_video`
- `frames_processed`
- `fps`
- `total_predictions`

Supported video formats are currently:

- `.mp4`
- `.avi`
- `.mov`

## Webcam Inference

Command:

```bash
uv run --python 3.11 python src/scripts/webcam.py --model models/yolo11n.pt --camera-index 0
```

Behavior:

- opens the requested camera index
- runs realtime inference on each frame
- overlays detections and FPS
- displays a window titled `YOLO Webcam Inference`

To stop:

- press `q`

Returned summary:

```json
{
  "success": true,
  "message": "Webcam inference stopped by user"
}
```

## Training

Training is driven by two YAML files:

- dataset config: `configs/data.yaml`
- train config: `configs/train.yaml`

### Validate Only

Use this first to verify paths and config values:

```bash
uv run --python 3.11 python src/scripts/train.py --validate-only
```

This confirms:

- dataset directories exist
- model path is resolvable
- training config values are valid

### Start Training

```bash
uv run --python 3.11 python src/scripts/train.py --data-config configs/data.yaml --train-config configs/train.yaml
```

Default training config:

- model: `models/yolo11n.pt`
- epochs: `10`
- imgsz: `640`
- batch: `16`
- project: `runs/train`
- name: `yolo11n-custom`
- device: `0`

When training starts successfully, output is written by YOLO to the configured project directory.

## Validation

Command:

```bash
uv run --python 3.11 python src/scripts/validate.py --data-config configs/data.yaml --train-config configs/train.yaml --split val
```

Supported splits:

- `train`
- `val`
- `test`

Returned metrics include:

- `precision`
- `recall`
- `map50`
- `map50_95`
- `speed_ms`

## Benchmarking

Command:

```bash
uv run --python 3.11 python src/scripts/benchmark.py --data-config configs/data.yaml --train-config configs/train.yaml --split test
```

This uses the same config flow as validation but returns benchmark-oriented output with:

- `mode: benchmark`
- evaluation metrics
- `benchmark_notes`

Repository benchmark helper scripts are also available:

```bash
uv run --python 3.11 python benchmarks/benchmark_inference.py --model models/yolo11n.pt
uv run --python 3.11 python benchmarks/benchmark_tracking.py --model models/yolo11n.pt
```

## Offline Tracking

Command:

```bash
uv run --python 3.11 python src/scripts/track.py outputs/sprint4_sample.mp4 --model models/yolo11n.pt
```

What it does:

- runs frame-by-frame detection
- matches detections across frames with IoU-based persistence
- draws object IDs and track history lines
- counts unique tracked people

Returned fields include:

- `people_count`
- `total_tracks`
- `frame_results`
- `output_video`

## API Usage

Start the API server:

```bash
uv run --python 3.11 uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Image prediction request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "outputs/sprint1_check_image.jpg",
    "model_source": "models/yolo11n.pt"
  }'
```

For the full contract, see [API_REFERENCE.md](/e:/YOLO/docs/API_REFERENCE.md).

## Docker Usage

Build:

```bash
docker build -t yolo-platform-api:1.0.0 .
```

Run standard profile:

```bash
docker compose up --build api
```

Run GPU profile:

```bash
docker compose --profile gpu up --build api-gpu
```

Mounted paths:

- `./models` to `/app/models`
- `./outputs` to `/app/outputs`

Health check endpoint inside the container:

```text
http://127.0.0.1:8000/api/v1/health
```

## Runtime Environment Variables

### Startup and Warmup

- `YOLO_STARTUP_VALIDATE_PATHS`
- `YOLO_WARMUP_ENABLED`
- `YOLO_WARMUP_MODEL`
- `YOLO_WARMUP_SKIP_IF_MISSING`
- `YOLO_MODELS_DIR`
- `YOLO_OUTPUTS_DIR`

### Logging

- `YOLO_LOG_LEVEL`
- `YOLO_LOG_JSON`
- `YOLO_LOG_FILE`

Examples:

```bash
export YOLO_WARMUP_ENABLED=false
export YOLO_LOG_JSON=true
```

## Troubleshooting

### Image not found

Check:

- the image path exists
- the path points to a file
- the image can be opened by PIL

### Video not found or unsupported

Supported formats are:

- `.mp4`
- `.avi`
- `.mov`

### Warmup model not found

Either:

- place the model under `models/`
- set `YOLO_WARMUP_MODEL` to a valid path
- or disable warmup with `YOLO_WARMUP_ENABLED=false`

### Webcam fails to open

Check:

- the camera index is correct
- the camera is not already in use
- OpenCV can access the device from the current environment

### Training config errors

Check:

- `configs/data.yaml`
- `configs/train.yaml`
- model path
- dataset directory structure

## Related Docs

- [README.md](/e:/YOLO/README.md)
- [API_REFERENCE.md](/e:/YOLO/docs/API_REFERENCE.md)
- [DATASET_GUIDE.md](/e:/YOLO/docs/DATASET_GUIDE.md)
- [DEPLOYMENT.md](/e:/YOLO/docs/DEPLOYMENT.md)
- [OPERATIONS.md](/e:/YOLO/docs/OPERATIONS.md)
