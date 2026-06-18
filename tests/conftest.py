"""Pytest configuration for repository-local imports and shared test assets."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = PROJECT_ROOT / "tests" / "fixtures"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    """Write a YAML file for test setup."""
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _create_dataset_layout(dataset_root: Path) -> None:
    """Create a minimal dataset directory structure for config validation."""
    for relative_path in (
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ):
        (dataset_root / relative_path).mkdir(parents=True, exist_ok=True)


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


@pytest.fixture
def training_validation_project(tmp_path: Path) -> dict[str, Path]:
    """Create isolated dataset, model, and config files for validation tests."""
    dataset_root = tmp_path / "dataset"
    _create_dataset_layout(dataset_root)

    model_path = tmp_path / "model.pt"
    model_path.write_text("model", encoding="utf-8")

    data_config_path = tmp_path / "data.yaml"
    train_config_path = tmp_path / "train.yaml"

    _write_yaml(
        data_config_path,
        {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": ["person"],
        },
    )
    _write_yaml(
        train_config_path,
        {
            "model": str(model_path),
            "epochs": 10,
            "imgsz": 640,
            "batch": 16,
            "project": str(tmp_path / "runs"),
            "name": "validation-test-run",
            "device": 0,
            "workers": 2,
            "patience": 20,
            "pretrained": True,
            "exist_ok": True,
            "verbose": True,
        },
    )

    return {
        "dataset_root": dataset_root,
        "model_path": model_path.resolve(),
        "data_config_path": data_config_path,
        "train_config_path": train_config_path,
        "project_dir": (tmp_path / "runs").resolve(),
    }
