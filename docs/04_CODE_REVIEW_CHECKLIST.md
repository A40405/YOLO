# CODE_REVIEW_CHECKLIST.md

# Mandatory Review Checklist

Before marking a sprint complete, verify every item below.

## Environment Validation

- [ ] All validation ran from WSL2 Ubuntu
- [ ] Working directory was `/mnt/e/YOLO`
- [ ] Python 3.11 was used
- [ ] `uv` was used where required
- [ ] No validation used Windows Python
- [ ] No validation used PowerShell
- [ ] No validation used CMD

## Architecture

- [ ] Code follows `ARCHITECTURE.md`
- [ ] Layer boundaries respected
- [ ] No circular imports
- [ ] No business logic in API layer
- [ ] No business logic in utils

## Dependencies

- [ ] API only depends on Services
- [ ] Services depend on Core
- [ ] Core does not depend on API
- [ ] Utils do not depend on Services

## Configuration

- [ ] No hardcoded paths
- [ ] No hardcoded thresholds
- [ ] No hardcoded model names
- [ ] Config loaded from yaml where the active sprint introduces config

## Logging

- [ ] Uses `loguru`
- [ ] Exceptions logged
- [ ] No `print()` in production code

## Error Handling

- [ ] Missing file handled
- [ ] Invalid image handled
- [ ] Missing model handled
- [ ] GPU unavailable handled

## Typing

- [ ] Public functions have type hints
- [ ] Public functions have docstrings

## YOLO Rules

- [ ] Models loaded via `ModelManager`
- [ ] No duplicate YOLO loading
- [ ] Inference standardized

## Testing

- [ ] Unit tests added when the active sprint introduces testable code
- [ ] Existing tests pass
- [ ] No broken imports

Run:

```bash
pytest
```

Only run this when the active sprint requires tests.

## API

- [ ] Swagger works
- [ ] Health endpoint works
- [ ] Predict endpoint works

## Docker

- [ ] Docker build succeeds
- [ ] Container starts successfully

Run:

```bash
docker build .
```

Only run this starting in Sprint 9.

## Acceptance

- [ ] Sprint goal achieved
- [ ] Validation passed
- [ ] Summary written
- [ ] Waiting for approval

Never continue to the next sprint automatically.
