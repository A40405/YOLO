"""Integration tests for training validation workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.training_service import TrainingService


pytestmark = pytest.mark.integration


def test_training_validate_only_uses_project_configs(training_validation_project: dict[str, Path]) -> None:
    """Ensure the training service validates training configs end to end."""
    service = TrainingService()

    summary = service.train(
        data_config_path=training_validation_project["data_config_path"],
        train_config_path=training_validation_project["train_config_path"],
        validate_only=True,
    )

    assert summary["success"] is True
    assert summary["mode"] == "validate-only"
    assert summary["model_source"] == str(training_validation_project["model_path"])
    assert summary["dataset_root"] == str(training_validation_project["dataset_root"])
    assert summary["class_count"] == 1
    assert summary["training_started"] is False
