"""Unit tests for YOLO validation and benchmarking flows."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.services.training_service import TrainingService


class FakeBoxMetrics:
    """Simple fake metrics payload for YOLO validation results."""

    def __init__(self, precision: float, recall: float, map50: float, map50_95: float) -> None:
        """Store scalar evaluation metrics."""
        self.mp = precision
        self.mr = recall
        self.map50 = map50
        self.map = map50_95


class FakeValidationResults:
    """Simple fake validation results object."""

    def __init__(self) -> None:
        """Initialize fake metrics and speed payload."""
        self.box = FakeBoxMetrics(precision=0.8765, recall=0.7654, map50=0.8123, map50_95=0.6543)
        self.speed = {"preprocess": 1.25, "inference": 7.5, "postprocess": 0.95}


class FakeYOLOModel:
    """Simple fake YOLO model for evaluation flows."""

    def __init__(self) -> None:
        """Initialize validation call storage."""
        self.val_calls: list[dict[str, object]] = []
        self.loaded_data_configs: list[dict[str, object]] = []

    def val(self, **kwargs: object) -> FakeValidationResults:
        """Record the validation call and return fake metrics."""
        self.val_calls.append(kwargs)
        data_config_path = Path(str(kwargs["data"]))
        self.loaded_data_configs.append(yaml.safe_load(data_config_path.read_text(encoding="utf-8")))
        return FakeValidationResults()


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


def _create_service(tmp_path: Path) -> tuple[TrainingService, FakeYOLOModel, Path, Path, Path]:
    """Create a configured training service with fake dependencies."""
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
            "name": "evaluation-run",
            "device": "cpu",
            "workers": 0,
            "patience": 5,
            "pretrained": True,
            "exist_ok": True,
            "verbose": False,
        },
    )

    fake_model = FakeYOLOModel()
    service = TrainingService(model_manager=FakeModelManager(fake_model))
    return service, fake_model, data_config, train_config, dataset_root


def test_validate_returns_expected_metrics_summary(tmp_path: Path) -> None:
    """Ensure validation returns the required metrics summary."""
    service, fake_model, data_config, train_config, dataset_root = _create_service(tmp_path)

    summary = service.validate(data_config, train_config, split="val")

    assert len(fake_model.val_calls) == 1
    assert summary["mode"] == "validate"
    assert summary["split"] == "val"
    assert summary["precision"] == 0.8765
    assert summary["recall"] == 0.7654
    assert summary["map50"] == 0.8123
    assert summary["map50_95"] == 0.6543
    assert summary["class_count"] == 2
    assert summary["dataset_root"] == str(dataset_root)
    assert summary["speed_ms"] == {"preprocess": 1.25, "inference": 7.5, "postprocess": 0.95}


def test_benchmark_returns_metrics_and_speed_notes(tmp_path: Path) -> None:
    """Ensure benchmarking returns the required metrics and benchmark notes."""
    service, fake_model, data_config, train_config, dataset_root = _create_service(tmp_path)

    summary = service.benchmark(data_config, train_config, split="test")

    assert len(fake_model.val_calls) == 1
    assert fake_model.val_calls[0]["split"] == "test"
    assert summary["mode"] == "benchmark"
    assert summary["split"] == "test"
    assert summary["benchmark_notes"] == "Speed metrics are reported in milliseconds per image."
    assert fake_model.loaded_data_configs == [
        {
            "path": str(dataset_root),
            "train": str((dataset_root / "images/train").resolve()),
            "val": str((dataset_root / "images/val").resolve()),
            "test": str((dataset_root / "images/test").resolve()),
            "names": ["person", "car"],
        }
    ]


def test_validate_rejects_invalid_split(tmp_path: Path) -> None:
    """Ensure invalid evaluation split names are rejected."""
    service, _, data_config, train_config, _ = _create_service(tmp_path)

    with pytest.raises(ValueError, match="Evaluation split must be one of"):
        service.validate(data_config, train_config, split="invalid")
