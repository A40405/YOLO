# Python Style Guide

- Python `3.11` only
- All commands must run from WSL2 Ubuntu at `/mnt/e/YOLO`
- Never use Windows Python, PowerShell, or CMD
- Full type hints on all public APIs
- Google-style docstrings
- Use `dataclasses` for structured data by default
- Use `Pydantic` only in sprints that introduce API or validation models
- No business logic in scripts
- Use `pathlib` instead of `os.path`
- Use logging, never `print()` in production
- Prefer composition over inheritance
- Max function length: 50 lines
- Max file length: 500 lines
- Mandatory `pytest` tests for service-layer code once services exist
