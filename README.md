# YOLO Platform

YOLO Platform is a production-oriented computer vision repository built on Ultralytics YOLO. Release `1.0.0` includes image inference, offline video inference, webcam inference, configuration-driven training and evaluation, a FastAPI service, Docker packaging, offline tracking, and test/coverage automation.

## Release Scope

Implemented in `1.0.0`:

- image inference
- offline video inference with annotated output
- webcam inference
- configuration-driven training
- validation and benchmarking
- FastAPI API with health and predict endpoints
- Docker and Docker Compose assets
- offline tracking with persistent IDs and people counting
- structured logging and startup validation
- unit, integration, and end-to-end tests

Not implemented in `1.0.0`:

- RTSP inference
- authentication and authorization
- database persistence
- PostgreSQL integration

## Architecture

The project follows the repository architecture defined in the specifications:

```text
API
  ->
Services
  ->
Core
  ->
YOLO
```

Key folders:

```text
src/
  api/        FastAPI app, routes, lifespan, middleware
  services/   Inference, training, video, webcam, tracking orchestration
  core/       YOLO model loading and execution
  utils/      Pure helpers
  scripts/    CLI entry points

configs/      Training and dataset YAML files
data/         Sample dataset
models/       Local model files
outputs/      Generated videos and sample assets
tests/        Unit, integration, and E2E tests
docs/         Deployment, operations, and user/developer documentation
benchmarks/   Benchmark helper scripts
```

## Requirements

- WSL2 Ubuntu
- Python `3.11`
- `uv`
- a local YOLO model file for commands that use `models/yolo11n.pt`
- NVIDIA GPU when you intend to run GPU validation or GPU-backed training/inference

Runtime dependencies are declared in [pyproject.toml](/e:/YOLO/pyproject.toml). The repository currently pins:

- `ultralytics`
- `torch==2.6.0`
- `torchvision==0.21.0`
- `torchaudio==2.6.0`
- `opencv-python`
- `loguru`
- `PyYAML`
- `fastapi`
- `uvicorn[standard]`

## Installation

From the repository root:

```bash
uv sync --python 3.11
```

Optional environment validation:

```bash
uv run --python 3.11 python src/scripts/check_gpu.py
uv run --python 3.11 python src/scripts/check_yolo.py
```

`check_yolo.py` creates a sample image under `outputs/` and attempts a YOLO inference run.

## Quick Start

### Image Inference

```bash
uv run --python 3.11 python src/scripts/predict.py outputs/sprint1_check_image.jpg --model models/yolo11n.pt
```

Expected output is JSON:

```json
[
  {
    "class": "person",
    "confidence": 0.91,
    "bbox": [0.0, 0.0, 100.0, 100.0]
  }
]
```

The exact predictions depend on the model and input image.

### Video Inference

```bash
uv run --python 3.11 python src/scripts/predict_video.py outputs/sprint4_sample.mp4 --model models/yolo11n.pt
```

This command writes an annotated video to `outputs/<input-name>_annotated.<ext>`.

### Webcam Inference

```bash
uv run --python 3.11 python src/scripts/webcam.py --model models/yolo11n.pt --camera-index 0
```

- Press `q` to stop the webcam window.
- The overlay includes an FPS counter.

### Tracking

```bash
uv run --python 3.11 python src/scripts/track.py outputs/sprint4_sample.mp4 --model models/yolo11n.pt
```

The tracking pipeline writes an annotated video and returns structured JSON that includes:

- `people_count`
- `total_tracks`
- per-frame track results

## Training Workflow

Training is configuration-driven.

Default configs:

- dataset config: `configs/data.yaml`
- training config: `configs/train.yaml`

Validate configs and paths only:

```bash
uv run --python 3.11 python src/scripts/train.py --validate-only
```

Start training:

```bash
uv run --python 3.11 python src/scripts/train.py --data-config configs/data.yaml --train-config configs/train.yaml
```

Default training config values in `configs/train.yaml`:

- model: `models/yolo11n.pt`
- epochs: `10`
- image size: `640`
- batch size: `16`
- project directory: `runs/train`
- run name: `yolo11n-custom`
- device: `0`

The training service resolves paths, validates directories, and then calls YOLO training through the core layer.

## Evaluation Workflow

Validation:

```bash
uv run --python 3.11 python src/scripts/validate.py --data-config configs/data.yaml --train-config configs/train.yaml --split val
```

Benchmarking:

```bash
uv run --python 3.11 python src/scripts/benchmark.py --data-config configs/data.yaml --train-config configs/train.yaml --split test
```

Supported splits:

- `train`
- `val`
- `test`

Returned evaluation metrics include:

- `precision`
- `recall`
- `map50`
- `map50_95`
- `speed_ms`

## API Usage

Start the API:

```bash
uv run --python 3.11 uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Predict request example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "outputs/sprint1_check_image.jpg",
    "model_source": "models/yolo11n.pt"
  }'
```

The API currently exposes:

- `GET /api/v1/health`
- `POST /api/v1/predict`

For request and response details, see [API_REFERENCE.md](/e:/YOLO/docs/API_REFERENCE.md).

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

Mounted volumes:

- `./models -> /app/models`
- `./outputs -> /app/outputs`

See [docs/DEPLOYMENT.md](/e:/YOLO/docs/DEPLOYMENT.md) and [docs/USER_GUIDE.md](/e:/YOLO/docs/USER_GUIDE.md) for operational details.

## Runtime Configuration

Relevant environment variables exposed by the current codebase include:

- `YOLO_STARTUP_VALIDATE_PATHS`
- `YOLO_WARMUP_ENABLED`
- `YOLO_WARMUP_MODEL`
- `YOLO_WARMUP_SKIP_IF_MISSING`
- `YOLO_MODELS_DIR`
- `YOLO_OUTPUTS_DIR`
- `YOLO_LOG_LEVEL`
- `YOLO_LOG_JSON`
- `YOLO_LOG_FILE`

The API startup lifecycle validates required directories and can optionally warm a model into the cache.

## Sample Dataset

The repository includes a sample dataset under `data/sample_dataset` with:

- `images/train`
- `images/val`
- `images/test`
- `labels/train`
- `labels/val`
- `labels/test`

Dataset configuration is documented in [DATASET_GUIDE.md](/e:/YOLO/docs/DATASET_GUIDE.md).

## Testing

Run the full test suite:

```bash
uv run --python 3.11 python -m pytest
```

Run tests with coverage:

```bash
uv run --python 3.11 python -m pytest --cov=src --cov-report=term-missing --cov-report=xml
```

Current coverage configuration:

- source: `src`
- scripts excluded from coverage
- minimum coverage: `80%`

## Benchmarking

Repository benchmark helper scripts:

```bash
uv run --python 3.11 python benchmarks/benchmark_inference.py --model models/yolo11n.pt
uv run --python 3.11 python benchmarks/benchmark_tracking.py --model models/yolo11n.pt
```

Benchmark JSON output should be compared with:

- `docs/BENCHMARK_RESULTS_TEMPLATE.md`

## Documentation Index

- [docs/USER_GUIDE.md](/e:/YOLO/docs/USER_GUIDE.md)
- [docs/API_REFERENCE.md](/e:/YOLO/docs/API_REFERENCE.md)
- [docs/DATASET_GUIDE.md](/e:/YOLO/docs/DATASET_GUIDE.md)
- [docs/DEPLOYMENT.md](/e:/YOLO/docs/DEPLOYMENT.md)
- [docs/OPERATIONS.md](/e:/YOLO/docs/OPERATIONS.md)

## Release Metadata

- Version: [VERSION](/e:/YOLO/VERSION)
- Changelog: [CHANGELOG.md](/e:/YOLO/CHANGELOG.md)
- Release notes: [RELEASE_NOTES_v1.0.0.md](/e:/YOLO/RELEASE_NOTES_v1.0.0.md)
