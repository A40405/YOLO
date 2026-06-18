# Deployment Guide

## Overview

This document describes how to deploy YOLO Platform `1.0.0` in the supported environment:

- WSL2 Ubuntu
- Python `3.11`
- `uv`
- NVIDIA GPU when GPU execution is required

Repository root:

```bash
<repo-root>
```

## WSL2 Setup

1. Install WSL2 with an Ubuntu distribution.
2. Ensure the repository is available locally and open a shell at the repository root.
3. Confirm the shell runs inside Ubuntu WSL2, not PowerShell or CMD.

## Python Setup

Use Python `3.11` only.

```bash
cd <repo-root>
uv python list
uv sync --python 3.11
```

## uv Setup

Install `uv` in WSL2 Ubuntu, then sync the repository:

```bash
cd <repo-root>
uv sync --python 3.11
```

## Docker Setup

1. Install Docker Desktop or Docker Engine with WSL2 integration enabled.
2. Confirm Docker is visible from the active WSL2 distro:

```bash
docker version
docker compose version
```

## GPU Setup

1. Install an NVIDIA driver on Windows that supports WSL2 GPU compute.
2. Install Docker GPU support if container GPU execution is required.
3. Confirm CUDA visibility from the project environment:

```bash
cd <repo-root>
uv run --python 3.11 python src/scripts/check_gpu.py
```

## API Startup

Run the API directly from WSL2 Ubuntu:

```bash
cd <repo-root>
uv run --python 3.11 uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

## Docker Deployment

Build the image:

```bash
cd <repo-root>
docker build -t yolo-platform-api:1.0.0 .
```

Run with Docker Compose:

```bash
cd <repo-root>
docker compose up --build api
```

Run GPU profile:

```bash
cd <repo-root>
docker compose --profile gpu up --build api-gpu
```

## Troubleshooting

### Docker not found in WSL2

- Enable WSL integration in Docker Desktop.
- Reopen the Ubuntu shell and re-run `docker version`.

### Model path errors

- Confirm models exist under `<repo-root>/models`.
- Confirm mounted Docker volumes expose `/app/models`.

### API startup validation fails

- Confirm `models/` and `outputs/` directories exist.
- Review startup environment variables such as `YOLO_MODELS_DIR` and `YOLO_OUTPUTS_DIR`.

### Warmup failure

- Disable warmup temporarily:

```bash
export YOLO_WARMUP_ENABLED=false
```

- Or point `YOLO_WARMUP_MODEL` to an existing model file.

### Low benchmark throughput

- Confirm GPU availability with `check_gpu.py`.
- Re-run benchmarks when no competing GPU workloads are active.
