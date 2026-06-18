# MASTER_SPEC.md

# YOLO Platform - Master Specification

## Objective

Build a production-ready computer vision platform based on Ultralytics YOLO.

The platform roadmap includes:

- Object detection
- Image inference
- Video inference
- Webcam inference
- RTSP inference
- Model training
- FastAPI service
- Docker deployment
- Future tracking support
- Future PostgreSQL integration

## Environment Policy

The only supported development and validation environment is:

- WSL2 Ubuntu
- Repository root: the checked-out project root
- Python `3.11`
- `uv` package manager
- NVIDIA GPU when a sprint requires GPU validation

Mandatory rules:

- All commands must be executed from WSL2 Ubuntu.
- Always run commands from the repository root.
- Never use Windows Python.
- Never use PowerShell.
- Never use CMD.
- Never validate using Windows interpreters.

## Validation Policy

Unless a sprint explicitly states otherwise, all validation commands assume:

- Ubuntu WSL2
- Python `3.11`
- `uv`
- Repository root

Validation must be performed in the active sprint only.

## Global Rules

IMPORTANT:

- Work sprint-by-sprint.
- Never implement future sprints.
- This file is the single source of truth.
- If another document conflicts with this file, this file wins.
- Release `1.0.0` is not complete until Sprint 13 documentation requirements are fully satisfied.

After each sprint:

1. Run validation.
2. Summarize changes.
3. Review against `docs/04_CODE_REVIEW_CHECKLIST.md`.
4. Stop and wait for approval.

Never continue automatically.

## Sprint 0 - Bootstrap

### Goal

Create repository foundation and environment configuration.

### Responsibilities

Sprint 0 is bootstrap only.

Allowed work:

- Create the repository structure
- Create `pyproject.toml`
- Create `README.md`
- Create `.gitignore`
- Configure `uv`
- Install and resolve dependencies
- Create a working virtual environment

Not allowed:

- YOLO code
- GPU checks
- FastAPI code
- Training code
- Database code
- Application scripts
- Business logic

Installing dependencies is allowed.
Writing application implementation code is not allowed.

### Deliverables

Repository structure:

```text
YOLO/
- configs/
- data/
- models/
- outputs/
- runs/
- src/
- tests/
- docker/
- README.md
- .gitignore
- pyproject.toml
```

### Runtime Dependencies

Sprint 0 MUST include these runtime dependencies in `pyproject.toml`:

- `ultralytics`
- `torch`
- `torchvision`
- `torchaudio`
- `opencv-python`
- `loguru`

### Development Dependencies

Sprint 0 MUST include these development dependencies in `pyproject.toml`:

- `pytest>=8.4.1`

### Required pyproject.toml

```toml
[project]
name = "yolo-platform"
version = "0.1.0"
description = "Production-ready YOLO platform"
readme = "README.md"
requires-python = ">=3.11,<3.12"

dependencies = [
    "ultralytics",
    "torch",
    "torchvision",
    "torchaudio",
    "opencv-python",
    "loguru"
]

[dependency-groups]
dev = [
    "pytest>=8.4.1"
]

[tool.uv]
package = false
```

### Acceptance Criteria

Must pass from WSL2 Ubuntu at the repository root:

```bash
uv sync
uv run python -c "print('ok')"
```

Must show:

- repository tree
- `pyproject.toml`
- validation results

STOP and wait for approval.

## Sprint 1 - Environment Validation

### Goal

Verify GPU and YOLO environment.

### Deliverables

- `src/scripts/check_gpu.py`
- `src/scripts/check_yolo.py`

### Acceptance Criteria

Verify:

- `torch.cuda.is_available()`
- GPU name
- YOLO model download
- Sample inference

STOP and wait for approval.

## Sprint 2 - Core Architecture

### Goal

Create core abstractions.

### Deliverables

`src/core/`

- `detector.py`
- `model_manager.py`

### Acceptance Criteria

- model caching
- singleton behavior
- no duplicate model loading

STOP and wait for approval.

## Sprint 3 - Image Inference

### Goal

Image prediction only.

### Deliverables

- `inference_service.py`
- `predict.py`

### Output Schema

```json
{
  "class": "person",
  "confidence": 0.91,
  "bbox": [0, 0, 100, 100]
}
```

STOP and wait for approval.

## Sprint 4 - Video Inference

### Deliverables

- `video_utils.py`
- `predict_video.py`

STOP and wait for approval.

## Sprint 5 - Webcam

### Deliverables

- `webcam.py`

STOP and wait for approval.

## Sprint 6 - Training Pipeline

### Deliverables

- `training_service.py`
- `train.py`
- `configs/data.yaml`
- `configs/train.yaml`

STOP and wait for approval.

## Sprint 7 - Evaluation

### Deliverables

- `validate.py`
- `benchmark.py`

### Metrics

- Precision
- Recall
- mAP50
- mAP50-95

STOP and wait for approval.

## Sprint 8 - FastAPI

### Deliverables

`src/api/`

### Endpoints

- `GET /health`
- `POST /predict`

STOP and wait for approval.

## Sprint 9 - Docker

### Deliverables

- `Dockerfile`
- `docker-compose.yml`

STOP and wait for approval.

## Sprint 10 - Tracking

### Deliverables

- `tracker.py`

### Features

- Object IDs
- Track history
- People counting

STOP and wait for approval.

## Sprint 11 - PostgreSQL

### Deliverables

- database layer
- repositories layer

STOP and wait for approval.

## Sprint 12 - Production Hardening

### Deliverables

- logging
- health checks
- configuration management
- error handling

Application must survive:

- invalid uploads
- missing models
- GPU unavailable
- invalid configs

STOP and wait for approval.

## Sprint 13 - Documentation & Developer Experience

### Goal

Create production-quality documentation that makes the platform understandable, usable, and maintainable for new developers and end users before release.

### Deliverables

Repository documentation:

- README.md
- docs/USER_GUIDE.md
- docs/API_REFERENCE.md
- docs/DATASET_GUIDE.md

### README Requirements

README must explain:

- project purpose
- key features
- architecture overview
- repository structure
- installation
- quick start
- image inference
- video inference
- webcam inference
- training workflow
- evaluation workflow
- API usage
- Docker usage
- testing
- benchmarking

README must be understandable by a new developer without reading source code.

### USER_GUIDE Requirements

Must explain:

- how to run the platform
- how to train a model
- how to evaluate a model
- how to run API inference
- how to run Docker deployment

### API_REFERENCE Requirements

Must document:

- endpoints
- request schemas
- response schemas
- example requests
- example responses

### DATASET_GUIDE Requirements

Must document:

- dataset structure
- YOLO label format
- train/val/test organization
- configuration examples

### Release Gate

Release `1.0.0` cannot be considered complete unless:

- all Sprint 13 deliverables exist in the repository
- the documentation matches the implemented platform behavior
- the documentation is sufficient for developer onboarding and basic user operation

### Acceptance Criteria

A new developer must be able to:

- understand the platform purpose
- run inference
- train a model
- call the API
- prepare a dataset in the expected format

using documentation only.

STOP and wait for approval.
