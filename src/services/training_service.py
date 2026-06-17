"""Training service for configuration-driven YOLO workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict

import yaml
from loguru import logger

from src.core.model_manager import ModelManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetConfig:
    """Resolved dataset configuration for YOLO training."""

    config_path: Path
    dataset_root: Path
    train_images: Path
    val_images: Path
    test_images: Path | None
    names: list[str]


@dataclass(frozen=True)
class TrainConfig:
    """Resolved training configuration for YOLO training."""

    config_path: Path
    model_source: str
    epochs: int
    imgsz: int
    batch: int
    project: Path
    name: str
    device: str | int
    workers: int
    patience: int
    pretrained: bool
    exist_ok: bool
    verbose: bool


class TrainingSummary(TypedDict):
    """Summary payload returned after validation or training orchestration."""

    success: bool
    message: str
    mode: str
    data_config: str
    train_config: str
    model_source: str
    dataset_root: str
    train_images: str
    val_images: str
    test_images: str | None
    class_count: int
    project_dir: str
    run_name: str
    epochs: int
    imgsz: int
    batch: int
    device: str | int
    training_started: bool
    training_result: str | None


class EvaluationSummary(TypedDict):
    """Summary payload returned after YOLO evaluation or benchmarking."""

    success: bool
    message: str
    mode: str
    split: str
    data_config: str
    train_config: str
    model_source: str
    dataset_root: str
    class_count: int
    run_name: str
    precision: float
    recall: float
    map50: float
    map50_95: float
    speed_ms: dict[str, float]
    benchmark_notes: str | None


class TrainingService:
    """Service layer orchestration for configuration-driven YOLO training."""

    def __init__(self, model_manager: ModelManager | None = None) -> None:
        """Initialize the service with a centralized model manager."""
        self._model_manager = model_manager or ModelManager()

    def train(
        self,
        data_config_path: str | Path,
        train_config_path: str | Path,
        *,
        validate_only: bool = False,
    ) -> TrainingSummary:
        """Validate configs and optionally start a YOLO training run."""
        dataset_config = self.load_data_config(data_config_path)
        train_config = self.load_train_config(train_config_path)
        summary = self._build_summary(dataset_config, train_config, validate_only=validate_only)

        if validate_only:
            logger.info("Training configuration validation completed successfully")
            return summary

        try:
            model = self._model_manager.get_model(train_config.model_source)
            with TemporaryDirectory(prefix="yolo-train-config-") as temp_dir:
                resolved_data_config_path = self._write_resolved_data_config(dataset_config, Path(temp_dir))
                train_results = model.train(
                    data=str(resolved_data_config_path),
                    epochs=train_config.epochs,
                    imgsz=train_config.imgsz,
                    batch=train_config.batch,
                    project=str(train_config.project),
                    name=train_config.name,
                    device=train_config.device,
                    workers=train_config.workers,
                    patience=train_config.patience,
                    pretrained=train_config.pretrained,
                    exist_ok=train_config.exist_ok,
                    verbose=train_config.verbose,
                )
        except Exception as exc:
            logger.exception("Training failed: {}", exc)
            raise

        summary["training_started"] = True
        summary["mode"] = "train"
        summary["message"] = "Training completed successfully."
        summary["training_result"] = str(train_results)
        logger.info("Training completed for run {}", train_config.name)
        return summary

    def validate(
        self,
        data_config_path: str | Path,
        train_config_path: str | Path,
        *,
        split: str = "val",
    ) -> EvaluationSummary:
        """Run YOLO validation and return core evaluation metrics."""
        return self._evaluate(
            data_config_path=data_config_path,
            train_config_path=train_config_path,
            split=split,
            mode="validate",
        )

    def benchmark(
        self,
        data_config_path: str | Path,
        train_config_path: str | Path,
        *,
        split: str = "test",
    ) -> EvaluationSummary:
        """Run YOLO benchmarking on a configured evaluation split."""
        return self._evaluate(
            data_config_path=data_config_path,
            train_config_path=train_config_path,
            split=split,
            mode="benchmark",
        )

    def load_data_config(self, config_path: str | Path) -> DatasetConfig:
        """Load and validate a dataset config file."""
        resolved_config_path = self._resolve_existing_file(config_path, label="Data config")
        config_data = self._load_yaml(resolved_config_path)

        path_value = config_data.get("path")
        train_value = config_data.get("train")
        val_value = config_data.get("val")
        test_value = config_data.get("test")
        names_value = config_data.get("names")

        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError("Data config must define a non-empty 'path' string.")
        if not isinstance(train_value, str) or not train_value.strip():
            raise ValueError("Data config must define a non-empty 'train' string.")
        if not isinstance(val_value, str) or not val_value.strip():
            raise ValueError("Data config must define a non-empty 'val' string.")
        if not isinstance(names_value, list) or not names_value or not all(
            isinstance(name, str) and name.strip() for name in names_value
        ):
            raise ValueError("Data config must define a non-empty 'names' list of class names.")

        dataset_root = self._resolve_config_path(path_value, base_path=resolved_config_path.parent)
        train_images = self._resolve_config_path(train_value, base_path=dataset_root)
        val_images = self._resolve_config_path(val_value, base_path=dataset_root)
        test_images = (
            self._resolve_config_path(test_value, base_path=dataset_root)
            if isinstance(test_value, str) and test_value.strip()
            else None
        )

        self._ensure_directory_exists(dataset_root, label="Dataset root")
        self._ensure_directory_exists(train_images, label="Train images directory")
        self._ensure_directory_exists(val_images, label="Validation images directory")
        if test_images is not None:
            self._ensure_directory_exists(test_images, label="Test images directory")

        logger.info("Validated dataset config {}", resolved_config_path)
        return DatasetConfig(
            config_path=resolved_config_path,
            dataset_root=dataset_root,
            train_images=train_images,
            val_images=val_images,
            test_images=test_images,
            names=[name.strip() for name in names_value],
        )

    def load_train_config(self, config_path: str | Path) -> TrainConfig:
        """Load and validate a training config file."""
        resolved_config_path = self._resolve_existing_file(config_path, label="Training config")
        config_data = self._load_yaml(resolved_config_path)

        model_value = config_data.get("model")
        name_value = config_data.get("name")
        project_value = config_data.get("project")
        epochs_value = config_data.get("epochs")
        imgsz_value = config_data.get("imgsz")
        batch_value = config_data.get("batch")
        device_value = config_data.get("device", 0)
        workers_value = config_data.get("workers", 2)
        patience_value = config_data.get("patience", 20)
        pretrained_value = config_data.get("pretrained", True)
        exist_ok_value = config_data.get("exist_ok", True)
        verbose_value = config_data.get("verbose", True)

        if not isinstance(model_value, str) or not model_value.strip():
            raise ValueError("Training config must define a non-empty 'model' string.")
        if not isinstance(name_value, str) or not name_value.strip():
            raise ValueError("Training config must define a non-empty 'name' string.")
        if not isinstance(project_value, str) or not project_value.strip():
            raise ValueError("Training config must define a non-empty 'project' string.")

        epochs = self._validate_positive_int(epochs_value, field_name="epochs")
        imgsz = self._validate_positive_int(imgsz_value, field_name="imgsz")
        batch = self._validate_positive_int(batch_value, field_name="batch")
        workers = self._validate_non_negative_int(workers_value, field_name="workers")
        patience = self._validate_non_negative_int(patience_value, field_name="patience")
        device = self._validate_device(device_value)
        pretrained = self._validate_bool(pretrained_value, field_name="pretrained")
        exist_ok = self._validate_bool(exist_ok_value, field_name="exist_ok")
        verbose = self._validate_bool(verbose_value, field_name="verbose")

        resolved_model_source = self._normalize_model_source(model_value, base_path=resolved_config_path.parent)
        project_path = self._resolve_config_path(project_value, base_path=resolved_config_path.parent)
        project_path.mkdir(parents=True, exist_ok=True)

        logger.info("Validated training config {}", resolved_config_path)
        return TrainConfig(
            config_path=resolved_config_path,
            model_source=resolved_model_source,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project_path,
            name=name_value.strip(),
            device=device,
            workers=workers,
            patience=patience,
            pretrained=pretrained,
            exist_ok=exist_ok,
            verbose=verbose,
        )

    def _build_summary(
        self,
        dataset_config: DatasetConfig,
        train_config: TrainConfig,
        *,
        validate_only: bool,
    ) -> TrainingSummary:
        """Build a structured training summary for CLI output."""
        return {
            "success": True,
            "message": "Configuration validation completed successfully."
            if validate_only
            else "Training is ready to start.",
            "mode": "validate-only" if validate_only else "train",
            "data_config": str(dataset_config.config_path),
            "train_config": str(train_config.config_path),
            "model_source": train_config.model_source,
            "dataset_root": str(dataset_config.dataset_root),
            "train_images": str(dataset_config.train_images),
            "val_images": str(dataset_config.val_images),
            "test_images": str(dataset_config.test_images) if dataset_config.test_images is not None else None,
            "class_count": len(dataset_config.names),
            "project_dir": str(train_config.project),
            "run_name": train_config.name,
            "epochs": train_config.epochs,
            "imgsz": train_config.imgsz,
            "batch": train_config.batch,
            "device": train_config.device,
            "training_started": False,
            "training_result": None,
        }

    def _evaluate(
        self,
        data_config_path: str | Path,
        train_config_path: str | Path,
        *,
        split: str,
        mode: str,
    ) -> EvaluationSummary:
        """Run YOLO validation once and build an evaluation summary."""
        normalized_split = self._validate_split(split)
        dataset_config = self.load_data_config(data_config_path)
        train_config = self.load_train_config(train_config_path)

        try:
            model = self._model_manager.get_model(train_config.model_source)
            with TemporaryDirectory(prefix="yolo-eval-config-") as temp_dir:
                resolved_data_config_path = self._write_resolved_data_config(dataset_config, Path(temp_dir))
                validation_results = model.val(
                    data=str(resolved_data_config_path),
                    split=normalized_split,
                    imgsz=train_config.imgsz,
                    batch=train_config.batch,
                    device=train_config.device,
                    workers=train_config.workers,
                    verbose=train_config.verbose,
                )
        except Exception as exc:
            logger.exception("YOLO {} failed: {}", mode, exc)
            raise

        summary = self._build_evaluation_summary(
            validation_results=validation_results,
            dataset_config=dataset_config,
            train_config=train_config,
            split=normalized_split,
            mode=mode,
        )
        logger.info("YOLO {} completed for run {} on split {}", mode, train_config.name, normalized_split)
        return summary

    def _write_resolved_data_config(self, dataset_config: DatasetConfig, output_dir: Path) -> Path:
        """Write a temporary YOLO dataset config with absolute resolved paths."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "resolved_data.yaml"
        payload: dict[str, Any] = {
            "path": str(dataset_config.dataset_root),
            "train": str(dataset_config.train_images),
            "val": str(dataset_config.val_images),
            "names": dataset_config.names,
        }
        if dataset_config.test_images is not None:
            payload["test"] = str(dataset_config.test_images)

        output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return output_path

    @staticmethod
    def _load_yaml(config_path: Path) -> dict[str, Any]:
        """Load a YAML file into a dictionary."""
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            logger.exception("Invalid YAML in {}: {}", config_path, exc)
            raise ValueError(f"Invalid YAML file: {config_path}") from exc

        if not isinstance(loaded, dict):
            raise ValueError(f"YAML config must contain a mapping: {config_path}")
        return loaded

    @staticmethod
    def _build_evaluation_summary(
        *,
        validation_results: Any,
        dataset_config: DatasetConfig,
        train_config: TrainConfig,
        split: str,
        mode: str,
    ) -> EvaluationSummary:
        """Build a structured summary from YOLO validation metrics."""
        box_metrics = validation_results.box
        speed_payload = getattr(validation_results, "speed", {}) or {}
        speed_ms = {
            str(key): round(float(value), 4)
            for key, value in speed_payload.items()
            if isinstance(key, str) and isinstance(value, (int, float))
        }

        return {
            "success": True,
            "message": "Validation completed successfully."
            if mode == "validate"
            else "Benchmark completed successfully.",
            "mode": mode,
            "split": split,
            "data_config": str(dataset_config.config_path),
            "train_config": str(train_config.config_path),
            "model_source": train_config.model_source,
            "dataset_root": str(dataset_config.dataset_root),
            "class_count": len(dataset_config.names),
            "run_name": train_config.name,
            "precision": round(float(box_metrics.mp), 4),
            "recall": round(float(box_metrics.mr), 4),
            "map50": round(float(box_metrics.map50), 4),
            "map50_95": round(float(box_metrics.map), 4),
            "speed_ms": speed_ms,
            "benchmark_notes": "Speed metrics are reported in milliseconds per image."
            if mode == "benchmark"
            else None,
        }

    @staticmethod
    def _resolve_existing_file(path_value: str | Path, *, label: str) -> Path:
        """Resolve a required config file and ensure it exists."""
        resolved_path = Path(path_value)
        if not resolved_path.is_absolute():
            resolved_path = (PROJECT_ROOT / resolved_path).resolve()
        else:
            resolved_path = resolved_path.resolve()

        if not resolved_path.exists() or not resolved_path.is_file():
            raise FileNotFoundError(f"{label} not found: {resolved_path}")
        return resolved_path

    @staticmethod
    def _resolve_config_path(path_value: str, *, base_path: Path) -> Path:
        """Resolve a config-relative or absolute path."""
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate.resolve()
        return (base_path / candidate).resolve()

    @staticmethod
    def _ensure_directory_exists(path_value: Path, *, label: str) -> None:
        """Ensure a required directory exists."""
        if not path_value.exists() or not path_value.is_dir():
            raise FileNotFoundError(f"{label} not found: {path_value}")

    @staticmethod
    def _validate_positive_int(value: Any, *, field_name: str) -> int:
        """Validate a positive integer configuration value."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Training config field '{field_name}' must be a positive integer.")
        return value

    @staticmethod
    def _validate_non_negative_int(value: Any, *, field_name: str) -> int:
        """Validate a non-negative integer configuration value."""
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Training config field '{field_name}' must be a non-negative integer.")
        return value

    @staticmethod
    def _validate_bool(value: Any, *, field_name: str) -> bool:
        """Validate a boolean configuration value."""
        if not isinstance(value, bool):
            raise ValueError(f"Training config field '{field_name}' must be a boolean.")
        return value

    @staticmethod
    def _validate_device(value: Any) -> str | int:
        """Validate a YOLO device setting."""
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("Training config field 'device' must be a non-negative integer or non-empty string.")

    @staticmethod
    def _normalize_model_source(model_value: str, *, base_path: Path) -> str:
        """Normalize model source paths while still allowing named YOLO weights."""
        candidate = Path(model_value)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Model not found: {resolved}")
            return str(resolved)

        relative_candidate = (base_path / candidate).resolve()
        if relative_candidate.exists():
            return str(relative_candidate)

        project_candidate = (PROJECT_ROOT / candidate).resolve()
        if project_candidate.exists():
            return str(project_candidate)

        if any(separator in model_value for separator in ("/", "\\")):
            raise FileNotFoundError(f"Model not found: {project_candidate}")

        return model_value.strip()

    @staticmethod
    def _validate_split(split: str) -> str:
        """Validate an evaluation split name."""
        normalized_split = split.strip().lower()
        if normalized_split not in {"train", "val", "test"}:
            raise ValueError("Evaluation split must be one of: train, val, test.")
        return normalized_split
