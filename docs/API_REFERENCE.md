# API Reference

## Overview

The implemented `1.0.0` API is a FastAPI application exposed from:

```text
src.api.app:app
```

Base path:

```text
/api/v1
```

Implemented endpoints:

- `GET /api/v1/health`
- `POST /api/v1/predict`

## Application Behavior

The API registers:

- structured exception handlers
- request logging middleware
- startup/shutdown lifespan hooks

Every request receives an `X-Request-ID` response header. If the client supplies `X-Request-ID`, that value is reused. Otherwise, the API generates one.

## `GET /api/v1/health`

### Purpose

Return a simple health payload that confirms the service is responding.

### Request

No request body.

### Response

Status code:

- `200 OK`

Response body:

```json
{
  "success": true,
  "message": "ok"
}
```

## `POST /api/v1/predict`

### Purpose

Run image inference through the service layer and return standardized predictions.

### Request Schema

Content type:

```text
application/json
```

Body:

```json
{
  "image_path": "string",
  "model_source": "string"
}
```

Field rules:

- `image_path`: required, non-empty string
- `model_source`: required, non-empty string

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "outputs/sprint1_check_image.jpg",
    "model_source": "models/yolo11n.pt"
  }'
```

### Success Response

Status code:

- `200 OK`

Response body shape:

```json
{
  "success": true,
  "image_path": "outputs/sprint1_check_image.jpg",
  "model_source": "models/yolo11n.pt",
  "predictions": [
    {
      "class": "person",
      "confidence": 0.91,
      "bbox": [1.0, 2.0, 3.0, 4.0]
    }
  ]
}
```

Prediction item fields:

- `class`: YOLO class name
- `confidence`: float between `0.0` and `1.0`
- `bbox`: list of four floats in `xyxy` order

### Error Responses

#### Missing file

Status code:

- `404 Not Found`

Example:

```json
{
  "success": false,
  "message": "Image not found: /path/to/file.jpg"
}
```

#### Invalid input or invalid image

Status code:

- `400 Bad Request`

Example:

```json
{
  "success": false,
  "message": "Invalid image: outputs/bad.jpg"
}
```

#### Validation error

Status code:

- `422 Unprocessable Entity`

This is returned by FastAPI/Pydantic when the request body does not satisfy the schema, for example when `image_path` is missing or empty.

#### Unexpected server error

Status code:

- `500 Internal Server Error`

Example:

```json
{
  "success": false,
  "message": "Internal server error."
}
```

## OpenAPI

When the server is running, OpenAPI is available at:

```text
/openapi.json
```

FastAPI’s interactive docs are also available through the default FastAPI documentation routes for the running application.

## Startup Validation and Environment Variables

The API lifespan logic validates runtime directories and supports optional warmup.

Supported runtime environment variables:

- `YOLO_STARTUP_VALIDATE_PATHS`
- `YOLO_WARMUP_ENABLED`
- `YOLO_WARMUP_MODEL`
- `YOLO_WARMUP_SKIP_IF_MISSING`
- `YOLO_MODELS_DIR`
- `YOLO_OUTPUTS_DIR`

Logging-related environment variables:

- `YOLO_LOG_LEVEL`
- `YOLO_LOG_JSON`
- `YOLO_LOG_FILE`

## Logging

The API logs:

- request start
- request completion
- request duration
- exception details
- startup and shutdown events

If file logging is enabled, the default log file path is:

```text
logs/app.log
```

## Notes

- The API currently supports image inference only.
- Video inference, webcam inference, training, benchmarking, and tracking are exposed through CLI scripts and service classes, not API endpoints.
