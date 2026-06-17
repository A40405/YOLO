# Operations Guide

## Startup Procedure

Direct runtime:

```bash
cd /mnt/e/YOLO
uv sync --python 3.11
uv run --python 3.11 uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Container runtime:

```bash
cd /mnt/e/YOLO
docker compose up --build api
```

## Shutdown Procedure

Direct runtime:

- Stop the process with `Ctrl+C`.

Container runtime:

```bash
cd /mnt/e/YOLO
docker compose down
```

## Model Update Procedure

1. Copy the new model file into `/mnt/e/YOLO/models`.
2. Update any benchmark or runtime commands to reference the new model path when needed.
3. Re-run health and inference validation:

```bash
cd /mnt/e/YOLO
uv run --python 3.11 python src/scripts/check_yolo.py
uv run --python 3.11 python src/scripts/validate.py
```

4. Restart the API if warmup caching is enabled.

## Log Inspection

Application logs default to `logs/app.log` when file logging is enabled.

Inspect logs:

```bash
cd /mnt/e/YOLO
tail -f logs/app.log
```

JSON logging can be enabled with:

```bash
export YOLO_LOG_JSON=true
```

## Health Check Procedure

Direct API health check:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Docker health check:

```bash
docker compose ps
```

## Benchmark Procedure

Inference benchmark:

```bash
cd /mnt/e/YOLO
uv run --python 3.11 python benchmarks/benchmark_inference.py --model models/yolo11n.pt
```

Tracking benchmark:

```bash
cd /mnt/e/YOLO
uv run --python 3.11 python benchmarks/benchmark_tracking.py --model models/yolo11n.pt
```

Preserve the JSON output and compare it against the template in `docs/BENCHMARK_RESULTS_TEMPLATE.md`.

## Backup Recommendations

- Back up the `models/` directory before replacing production models.
- Back up `outputs/` when annotated outputs or benchmark artifacts must be retained.
- Back up `configs/` and release documentation before operational changes.
- Keep exported benchmark JSON reports with release notes for traceability.
