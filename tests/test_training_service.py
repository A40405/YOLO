"""Unit tests for the training service."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.services.training_service import TrainingService


class FakeYOLOModel:
    """Simple fake YOLO model used to verify training orchestration."""

    def __init__(self) -> None:
        """Initialize fake train call storage."""
        self.train_calls: list[dict[str, object]] = []
        self.loaded_data_configs: list[dict[str, object]] = []

    def train(self, **kwargs: object) -> str:
        """Record the training call and return a fake summary."""
        self.train_calls.append(kwargs)
        data_config_path = Path(str(kwargs["data"]))
        self.loaded_data_configs.append(yaml.safe_load(data_config_path.read_text(encoding="utf-8")))
        return "fake-training-result"


class FakeModelManager:
    """Simple fake model manager used to avoid real YOLO loading in tests."""

    def __init__(self, model: FakeYOLOModel) -> None:
        """Store the fake YOLO model instance."""
        self._model = model
        self.requested_sources: list[str] = []

    def get_model(self, model_source: str | Path) -> FakeYOLOModel:
        """Record the model source request and return the fake model."""
        self.requested_sources.append(str(model_source))
        return self._model


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


def test_validate_only_returns_training_summary(tmp_path: Path) -> None:
    """Ensure validate-only mode returns a structured summary without training."""
    dataset_root = tmp_path / "dataset"
    _create_dataset_layout(dataset_root)

    data_config = tmp_path / "data.yaml"
    train_config = tmp_path / "train.yaml"
    model_path = tmp_path / "model.pt"
    model_path.write_text("model", encoding="utf-8")

    _write_yaml(
        data_config,
        {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": ["person", "car"],
        },
    )
    _write_yaml(
        train_config,
        {
            "model": str(model_path),
            "epochs": 5,
            "imgsz": 640,
            "batch": 8,
            "project": str(tmp_path / "runs"),
            "name": "unit-test-run",
            "device": "cpu",
            "workers": 0,
            "patience": 5,
            "pretrained": True,
            "exist_ok": True,
            "verbose": False,
        },
    )

    model = FakeYOLOModel()
    service = TrainingService(model_manager=FakeModelManager(model))

    summary = service.train(data_config, train_config, validate_only=True)

    assert summary["success"] is True
    assert summary["mode"] == "validate-only"
    assert summary["training_started"] is False
    assert summary["class_count"] == 2
    assert model.train_calls == []


def test_train_uses_model_manager_and_resolved_data_config(tmp_path: Path) -> None:
    """Ensure training orchestration uses the cached model manager and resolved config paths."""
    dataset_root = tmp_path / "dataset"
    _create_dataset_layout(dataset_root)

    data_config = tmp_path / "data.yaml"
    train_config = tmp_path / "train.yaml"
    model_path = tmp_path / "model.pt"
    model_path.write_text("model", encoding="utf-8")

    _write_yaml(
        data_config,
        {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": ["person"],
        },
    )
    _write_yaml(
        train_config,
        {
            "model": str(model_path),
            "epochs": 3,
            "imgsz": 320,
            "batch": 4,
            "project": str(tmp_path / "runs"),
            "name": "fake-train-run",
            "device": 0,
            "workers": 1,
            "patience": 2,
            "pretrained": True,
            "exist_ok": True,
            "verbose": True,
        },
    )

    fake_model = FakeYOLOModel()
    fake_manager = FakeModelManager(fake_model)
    service = TrainingService(model_manager=fake_manager)

    summary = service.train(data_config, train_config, validate_only=False)

    assert fake_manager.requested_sources == [str(model_path)]
    assert len(fake_model.train_calls) == 1
    train_call = fake_model.train_calls[0]
    assert Path(str(train_call["data"])).name == "resolved_data.yaml"
    assert train_call["epochs"] == 3
    assert train_call["imgsz"] == 320
    assert train_call["batch"] == 4
    assert train_call["project"] == str((tmp_path / "runs").resolve())
    assert fake_model.loaded_data_configs == [
        {
            "path": str(dataset_root),
            "train": str((dataset_root / "images/train").resolve()),
            "val": str((dataset_root / "images/val").resolve()),
            "test": str((dataset_root / "images/test").resolve()),
            "names": ["person"],
        }
    ]
    assert summary["training_started"] is True
    assert summary["training_result"] == "fake-training-result"


def test_load_data_config_rejects_missing_dataset_directory(tmp_path: Path) -> None:
    """Ensure missing dataset directories are rejected clearly."""
    data_config = tmp_path / "data.yaml"
    _write_yaml(
        data_config,
        {
            "path": str(tmp_path / "missing-dataset"),
            "train": "images/train",
            "val": "images/val",
            "names": ["person"],
        },
    )

    service = TrainingService()

    with pytest.raises(FileNotFoundError, match="Dataset root not found"):
        service.load_data_config(data_config)


def test_load_train_config_rejects_missing_model_path(tmp_path: Path) -> None:
    """Ensure explicit missing model paths are rejected clearly."""
    train_config = tmp_path / "train.yaml"
    _write_yaml(
        train_config,
        {
            "model": "models/missing.pt",
            "epochs": 5,
            "imgsz": 640,
            "batch": 8,
            "project": str(tmp_path / "runs"),
            "name": "bad-run",
            "device": 0,
            "workers": 0,
            "patience": 5,
            "pretrained": True,
            "exist_ok": True,
            "verbose": True,
        },
    )

    service = TrainingService()

    with pytest.raises(FileNotFoundError, match="Model not found"):
        service.load_train_config(train_config)
