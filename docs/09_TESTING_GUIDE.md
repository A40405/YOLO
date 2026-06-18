# Testing Guide

- All validation must run from WSL2 Ubuntu at the repository root
- Use Python `3.11` with `uv`
- Never validate with Windows Python, PowerShell, or CMD
- Sprint 0 validation is bootstrap-only:
  - `uv sync`
  - `uv run python -c "print('ok')"`
- Use `pytest` for testable code sprints
- Mock YOLO in unit tests
- Integration tests for API apply starting in Sprint 8
- Coverage target `>= 80%` applies when meaningful application code exists
- CI must run tests before merge
- No GPU dependency in most tests
- Sprint 13 documentation must be reviewed as part of release readiness for `1.0.0`
