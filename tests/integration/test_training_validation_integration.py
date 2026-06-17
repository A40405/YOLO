"""Integration tests for training validation workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.training_service import TrainingService


pytestmark = pytest.mark.integration


def test_training_validate_only_uses_project_configs() -> None:
    """Ensure the training service validates the shared project configs end to end."""
    service = TrainingService()

    summary = service.train(
        data_config_path=Path("configs/data.yaml"),
        train_config_path=Path("configs/train.yaml"),
        validate_only=True,
    )

    assert summary["success"] is True
    assert summary["mode"] == "validate-only"
    assert summary["model_source"].endswith("models/yolo11n.pt")
    assert summary["dataset_root"].endswith("data/sample_dataset")
    assert summary["class_count"] == 1
    assert summary["training_started"] is False
