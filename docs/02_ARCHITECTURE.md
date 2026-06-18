# ARCHITECTURE.md

# YOLO Platform Architecture Guide

## Purpose

This document defines architecture, coding standards, folder ownership, dependency rules, and implementation constraints for AI coding agents.

The objective is consistency, maintainability, and minimizing hallucination during multi-phase development.

## Source Of Truth

`docs/01_MASTER_SPEC.md` is the single source of truth.

If this document conflicts with the master spec, the master spec wins.

## Environment Policy

All architecture work assumes:

- WSL2 Ubuntu only
- Repository root
- Python `3.11`
- `uv`

Never use Windows Python, PowerShell, or CMD for development or validation.

## Sprint Scope

This document defines architecture rules for the full project, but it does not authorize implementing future sprints early.

Use only the rules relevant to the active sprint.

- Sprint 0 does not include application code.
- Sprint 1 introduces environment validation scripts.
- Core, service, API, and deployment rules apply only when those sprints begin.

## Architecture Style

Use Clean Architecture.

Layers:

```text
API Layer
    ->
Service Layer
    ->
Core Layer
    ->
External Dependencies
```

Rules:

- API must never call YOLO directly.
- API must call Services.
- Services must call Core.
- Core owns model execution.
- Utilities must never contain business logic.

## Folder Ownership

### src/api

Responsibilities:

- FastAPI
- Request validation
- Response formatting
- Routing

Must NOT:

- Load models
- Run training
- Execute YOLO inference directly

### src/services

Responsibilities:

- Business workflows
- Orchestration
- Input/output transformation

Examples:

- InferenceService
- TrainingService
- ExportService

### src/core

Responsibilities:

- Model loading
- YOLO execution
- Tracking execution

Examples:

- Detector
- ModelManager
- Tracker

Only this layer may interact with Ultralytics directly.

### src/utils

Responsibilities:

- Pure helpers

Examples:

- image_utils.py
- video_utils.py
- logger.py

No business logic.

### tests

Mirror production structure.

Example:

```text
src/core/detector.py
tests/test_detector.py
```

## Dependency Rules

Allowed:

```text
API -> Services
Services -> Core
Services -> Utils
Core -> Utils
```

Forbidden:

```text
Core -> API
Core -> Services
Utils -> Services
Utils -> Core
```

## Model Management

All model loading must go through:

```text
ModelManager
```

Requirements:

- Singleton behavior
- Lazy loading
- Model cache

Never instantiate YOLO repeatedly.

Bad:

```python
YOLO("yolo11n.pt")
YOLO("yolo11n.pt")
YOLO("yolo11n.pt")
```

Good:

```python
manager.get_model("yolo11n.pt")
```

## Configuration Rules

Never hardcode:

- Paths
- Model names
- Thresholds
- Device IDs

Use:

```text
configs/*.yaml
```

Examples:

```yaml
confidence_threshold: 0.25
iou_threshold: 0.45
device: 0
```

This rule becomes active only in sprints that introduce config files.

## Logging Standards

Use:

```python
from loguru import logger
```

Files:

```text
logs/app.log
logs/train.log
logs/predict.log
```

These log files are relevant only once production code exists.
They are not a Sprint 0 deliverable.

Every exception must be logged.

Never use `print()` in production code.

Allowed only in scripts.

## Error Handling

Always catch:

- Missing files
- Missing models
- Invalid images
- GPU unavailable

Return structured errors.

Example:

```json
{
  "success": false,
  "message": "Model not found"
}
```

## API Standards

Base URL:

```text
/api/v1
```

Naming:

```text
GET /health
POST /predict
POST /train
```

Responses:

```json
{
  "success": true,
  "data": {}
}
```

Errors:

```json
{
  "success": false,
  "message": "error"
}
```

## Training Standards

Training code belongs in:

```text
src/services/training_service.py
```

Scripts only trigger services.

Bad:

```python
# train.py
YOLO(...).train(...)
```

Good:

```python
service.train()
```

## Inference Standards

Output schema:

```json
{
  "class": "person",
  "confidence": 0.91,
  "bbox": [0, 0, 100, 100]
}
```

Never expose raw Ultralytics objects outside Core layer.

## Testing Standards

Framework:

```text
pytest
```

Requirements:

- Unit tests
- No GPU dependency for most tests
- Mock model execution when possible

Target coverage:

```text
80%+
```

## Docker Standards

Dockerfile must:

- Use Python 3.11
- Use UV
- Run FastAPI

Startup:

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

This rule applies starting in Sprint 8 and later.

## Naming Conventions

Files:

```text
snake_case.py
```

Classes:

```text
PascalCase
```

Functions:

```text
snake_case
```

Constants:

```text
UPPER_CASE
```

## Code Quality Rules

Every public function must have:

- Type hints
- Docstring

Example:

```python
def predict_image(path: str) -> list:
    """Run inference on an image."""
```

## Agent Rules

IMPORTANT:

1. Implement only the current sprint.
2. Do not create future sprint code.
3. Run validation after implementation.
4. Explain how to test.
5. Stop and wait for approval.

Never continue automatically.

This rule overrides all other instructions.
