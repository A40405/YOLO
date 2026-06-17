# Release Notes - 1.0.0

## Overview

Release `1.0.0` marks the first production-ready milestone of the YOLO Platform.
The roadmap delivered object detection, image inference, video inference, webcam inference,
training, evaluation, FastAPI serving, Docker packaging, tracking, testing, and production hardening.

## Key Features

- Cached YOLO model management through `ModelManager`
- Image inference with standardized predictions
- Offline video inference with annotated outputs
- Webcam inference with live FPS display
- Configuration-driven training and evaluation
- FastAPI API with `/api/v1/health` and `/api/v1/predict`
- Docker and Docker Compose deployment assets
- Offline video tracking with persistent IDs, history trails, and people counting
- Integration, end-to-end, and coverage-based quality validation
- Startup validation, optional model warmup, structured logging, and request logging middleware

## Architecture Overview

The platform follows Clean Architecture:

- `src/api`: FastAPI routing, request validation, response formatting, lifespan, middleware
- `src/services`: orchestration for inference, training, evaluation, video processing, webcam, and tracking
- `src/core`: model loading and YOLO execution through `Detector` and `ModelManager`
- `src/utils`: pure helpers such as video utilities and logging configuration

YOLO interaction remains centralized in the core layer through `ModelManager` and `Detector`.
APIs call services only, and services own business workflows.

## Deployment Overview

- Python `3.11` with `uv`
- WSL2 Ubuntu development and validation environment
- FastAPI served with `uvicorn`
- Docker multi-stage image with health check support
- Docker Compose profiles for standard and GPU-capable runtime
- Models and outputs mounted as host volumes

## Known Limitations

- RTSP inference is not implemented in `1.0.0`
- No authentication or authorization layer is included
- No database persistence or PostgreSQL integration is included
- Tracking uses a lightweight IoU-based persistence strategy rather than a dedicated external tracker such as ByteTrack
- Webcam benchmarks depend on a locally available camera device
- Local Docker validation requires Docker integration to be available in the active WSL2 distro

## Future Roadmap

- RTSP streaming support
- Stronger tracking backends and analytics
- Database persistence and repository layers
- Expanded deployment automation
- Extended operational monitoring and alerting
