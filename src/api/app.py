"""FastAPI application entrypoint for the YOLO platform API."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.lifespan import app_lifespan
from src.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="YOLO Platform API", version="0.1.0", lifespan=app_lifespan)
    _register_request_logging_middleware(app)
    app.include_router(router)
    _register_exception_handlers(app)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Register structured exception handlers for the API."""

    @app.exception_handler(FileNotFoundError)
    async def handle_file_not_found(_: Request, exc: FileNotFoundError) -> JSONResponse:
        """Return a structured 404 response for missing files."""
        logger.exception("API file-not-found error: {}", exc)
        return JSONResponse(
            status_code=404,
            content=_build_error_response(message=str(exc)),
        )

    @app.exception_handler(ValueError)
    async def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
        """Return a structured 400 response for invalid user input."""
        logger.exception("API value error: {}", exc)
        return JSONResponse(
            status_code=400,
            content=_build_error_response(message=str(exc)),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        """Return a structured 500 response for unexpected failures."""
        logger.exception("API unexpected error: {}", exc)
        return JSONResponse(
            status_code=500,
            content=_build_error_response(message="Internal server error."),
        )


def _build_error_response(message: str) -> dict[str, Any]:
    """Build a standard structured error payload."""
    return {
        "success": False,
        "message": message,
    }


def _register_request_logging_middleware(app: FastAPI) -> None:
    """Register middleware that logs each request with timing information."""

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> JSONResponse | Any:
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        start_time = time.perf_counter()
        request_logger = logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        request_logger.info("Request started")

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            request_logger.bind(duration_ms=duration_ms).exception("Request failed")
            raise

        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        response.headers["X-Request-ID"] = request_id
        request_logger.bind(status_code=response.status_code, duration_ms=duration_ms).info("Request completed")
        return response


app = create_app()
