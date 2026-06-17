# TASKS.md

# YOLO Platform Development Backlog

## Working Rules

- `docs/01_MASTER_SPEC.md` is the single source of truth.
- Complete one sprint at a time.
- Do not start the next sprint until approved.
- All commands and validation must run from WSL2 Ubuntu at `/mnt/e/YOLO`.
- Use Python `3.11` and `uv`.
- Never use Windows Python, PowerShell, or CMD.
- Every sprint must have validation steps.
- Every sprint must end with a summary.

## Sprint 0 - Bootstrap

### Goal

Create repository foundation and environment configuration.

### Tasks

- Create folder structure
- Create `pyproject.toml`
- Create `README.md`
- Create `.gitignore`
- Configure `uv`
- Run `uv sync`

### Runtime Dependencies

- `ultralytics`
- `torch`
- `torchvision`
- `torchaudio`
- `opencv-python`
- `loguru`

### Development Dependencies

- `pytest>=8.4.1`

### Restrictions

- No YOLO code
- No GPU checks
- No FastAPI code
- No training code
- No database code
- No application scripts
- No business logic

### Deliverables

- Repository skeleton
- Working virtual environment
- Dependencies declared in `pyproject.toml`

### Validation

```bash
uv sync
uv run python -c "print('ok')"
```

Validation assumptions:

- WSL2 Ubuntu
- Repository root `/mnt/e/YOLO`
- Python `3.11`
- `uv`

## Sprint 1 - Environment Validation

### Goal

Verify GPU and YOLO setup.

### Tasks

- Implement `check_gpu.py`
- Implement `check_yolo.py`
- Verify CUDA
- Verify model download

### Validation

```bash
uv run src/scripts/check_gpu.py
uv run src/scripts/check_yolo.py
```

Expected:

- CUDA detected
- YOLO model downloaded
- Inference successful

## Sprint 2 - Core Architecture

### Goal

Build model abstraction layer.

### Tasks

- Create `ModelManager`
- Create `Detector`
- Add model cache
- Add singleton pattern

### Validation

- Model loads only once
- Cached model reused

## Sprint 3 - Image Inference

### Goal

Support image prediction.

### Tasks

- Create `InferenceService`
- Create `predict.py`
- Standardize output schema

### Validation

```bash
uv run src/scripts/predict.py image.jpg
```

## Sprint 4 - Video Inference

### Goal

Support offline video processing.

### Tasks

- Implement `video_utils`
- Implement `predict_video.py`
- Save annotated video

### Validation

- Output video generated

## Sprint 5 - Webcam

### Goal

Realtime inference.

### Tasks

- Implement `webcam.py`
- Display FPS
- Display detections

### Validation

- Webcam stream works

## Sprint 6 - Training

### Goal

Support custom dataset training.

### Tasks

- Create `training_service.py`
- Create `train.py`
- Create `configs/data.yaml`
- Create `configs/train.yaml`

### Validation

```bash
uv run src/scripts/train.py
```

## Sprint 7 - Evaluation

### Goal

Evaluate trained models.

### Tasks

- `validate.py`
- `benchmark.py`

Metrics:

- Precision
- Recall
- mAP50
- mAP50-95

### Validation

- Validation metrics are produced and reported

## Sprint 8 - API

### Goal

Serve inference via FastAPI.

### Tasks

- Create API structure
- Create health endpoint
- Create predict endpoint

### Validation

- Swagger available

## Sprint 9 - Docker

### Goal

Containerization.

### Tasks

- `Dockerfile`
- `docker-compose.yml`

### Validation

```bash
docker build .
```

## Sprint 10 - Tracking

### Goal

ByteTrack integration.

### Tasks

- `tracker.py`
- object IDs
- people counting

### Validation

- Tracking IDs are stable in sample output

## Sprint 11 - Database

### Goal

Persist detections.

### Tasks

- PostgreSQL integration
- repositories layer
- migrations

### Validation

- Database integration works with configured storage

## Sprint 12 - Production

### Goal

Production readiness.

### Tasks

- Logging
- Error handling
- Health checks
- Config management
- Deployment preparation

### Validation

- Required resilience scenarios are verified
