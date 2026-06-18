"""Pytest configuration for repository-local imports and shared test assets."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = PROJECT_ROOT / "tests" / "fixtures"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    """Return the checked-in sample image fixture used by end-to-end tests."""
    return FIXTURES_ROOT / "images" / "sample_image.png"


@pytest.fixture(scope="session")
def yolo_model_source() -> str:
    """Resolve a local model path when available, otherwise use the named YOLO weight."""
    local_model_path = PROJECT_ROOT / "models" / "yolo11n.pt"
    if local_model_path.exists():
        return str(local_model_path)
    return "yolo11n.pt"
