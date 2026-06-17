"""Model loading and caching utilities for YOLO models."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from loguru import logger
from ultralytics import YOLO


class ModelManager:
    """Singleton-style manager that lazily loads and caches YOLO models."""

    _instance: ModelManager | None = None
    _instance_lock: Lock = Lock()

    def __new__(cls) -> ModelManager:
        """Create or return the singleton model manager instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize cache state once for the singleton instance."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._cache: dict[str, YOLO] = {}
        self._cache_lock = Lock()
        self._initialized = True

    def get_model(self, model_source: str | Path) -> YOLO:
        """Return a cached YOLO model, loading it on first access."""
        cache_key = self._build_cache_key(model_source)

        with self._cache_lock:
            cached_model = self._cache.get(cache_key)
            if cached_model is not None:
                logger.info("Reusing cached YOLO model for {}", cache_key)
                return cached_model

            try:
                logger.info("Loading YOLO model for {}", cache_key)
                model = YOLO(cache_key)
                self._cache[cache_key] = model
                return model
            except Exception as exc:
                logger.exception("Failed to load YOLO model {}: {}", cache_key, exc)
                raise

    def clear_cache(self) -> None:
        """Clear the in-memory model cache."""
        with self._cache_lock:
            self._cache.clear()
            logger.info("Cleared YOLO model cache")

    def cache_size(self) -> int:
        """Return the current number of cached models."""
        with self._cache_lock:
            return len(self._cache)

    @staticmethod
    def _build_cache_key(model_source: str | Path) -> str:
        """Build a stable cache key for local or remote model sources."""
        source_path = Path(model_source)
        if source_path.exists():
            return str(source_path.resolve())
        return str(model_source)
