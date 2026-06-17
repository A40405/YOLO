# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Release preparation guidance, CI/CD automation, benchmarking assets, and deployment documentation will evolve here after `1.0.0`.

## [1.0.0] - 2026-06-17

### Added
- Sprint 0 bootstrap foundation with repository structure, `uv` configuration, and Python 3.11 dependency management.
- Sprint 1 environment validation scripts for CUDA availability, GPU discovery, YOLO model download, and sample inference.
- Sprint 2 core model abstractions with `ModelManager` singleton caching and `Detector` wrappers.
- Sprint 3 image inference workflow with standardized prediction output.
- Sprint 4 offline video inference with annotated output video generation.
- Sprint 5 webcam inference with realtime display and FPS overlay.
- Architecture cleanup with dedicated video and webcam services while preserving behavior.
- Sprint 6 training pipeline with configuration-driven dataset validation and YOLO training orchestration.
- Sprint 7 evaluation and benchmarking scripts for validation metrics including Precision, Recall, `mAP50`, and `mAP50-95`.
- Sprint 8 FastAPI serving layer with `GET /api/v1/health` and `POST /api/v1/predict`.
- Sprint 9 Docker deployment assets with GPU-ready compose profiles and health checks.
- Sprint 10 offline object tracking with persistent IDs, history trails, and people counting.
- Sprint 11 integration, end-to-end, and coverage-based quality gates.
- Sprint 12 production hardening with lifespan startup validation, optional model warmup, structured logging, and request logging middleware.
- Release 1.0.0 preparation assets including changelog, release notes, CI/CD workflows, benchmark scripts, and deployment operations documentation.

### Changed
- Project version updated from `0.1.0` to `1.0.0`.
- Test tooling extended with coverage reporting and benchmark documentation templates.

### Fixed
- Repeated YOLO model loading was prevented through centralized `ModelManager` caching.
- API, training, and tracking workflows now share stronger validation and observability behavior from prior hardening work.
