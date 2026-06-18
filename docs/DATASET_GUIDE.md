# Dataset Guide

## Purpose

This guide documents the dataset structure expected by the implemented training and evaluation workflows in release `1.0.0`.

The training and evaluation commands use:

- `configs/data.yaml`
- `configs/train.yaml`

The current sample dataset lives under:

```text
data/sample_dataset
```

## Current Dataset Config

The repository’s default dataset config is:

```yaml
path: ../data/sample_dataset
train: images/train
val: images/val
test: images/test
names:
  - person
```

Interpretation:

- `path` is the dataset root
- `train`, `val`, and `test` are relative to that dataset root
- `names` is the ordered class-name list

## Required Directory Structure

The implemented training service expects the configured directories to exist.

Sample structure:

```text
data/sample_dataset/
  images/
    train/
      sample_train.jpg
    val/
      sample_val.jpg
    test/
      sample_test.jpg
  labels/
    train/
      sample_train.txt
    val/
      sample_val.txt
    test/
      sample_test.txt
```

The repository also currently contains YOLO cache files under `labels/`:

- `labels/val.cache`
- `labels/test.cache`

## Image and Label Pairing

For each image file in a split, the matching label file should:

- be in the corresponding `labels/<split>/` directory
- share the same stem name

Example:

```text
images/train/sample_train.jpg
labels/train/sample_train.txt
```

## YOLO Label Format

Label files use the standard YOLO text format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Where:

- `class_id` is a zero-based integer index into `names`
- `x_center`, `y_center`, `width`, and `height` are normalized values in the range `0.0` to `1.0`

Example for the single class `person`:

```text
0 0.500000 0.500000 0.250000 0.400000
```

Because the default config has:

```yaml
names:
  - person
```

the only valid class index in the sample dataset is:

```text
0
```

## Train/Val/Test Organization

The implemented config and services support these splits:

- `train`
- `val`
- `test`

Training uses:

- `train`
- `val`

Validation and benchmarking support:

- `train`
- `val`
- `test`

The training service validates that:

- the dataset root exists
- the train image directory exists
- the validation image directory exists
- the test image directory exists if configured

## Configuration Rules

The training service requires these fields in the dataset config:

- `path`: non-empty string
- `train`: non-empty string
- `val`: non-empty string
- `names`: non-empty list of strings

Optional:

- `test`

If any required field is missing or malformed, the training service raises a validation error.

## Example Custom Dataset Config

Example with two classes:

```yaml
path: ../data/my_dataset
train: images/train
val: images/val
test: images/test
names:
  - person
  - bicycle
```

In this case:

- `person` uses class index `0`
- `bicycle` uses class index `1`

## Training Config Relationship

The default training config is:

```yaml
model: models/yolo11n.pt
epochs: 10
imgsz: 640
batch: 16
project: ../runs/train
name: yolo11n-custom
device: 0
workers: 2
patience: 20
pretrained: true
exist_ok: true
verbose: true
```

The dataset config and train config are used together by:

- `src/scripts/train.py`
- `src/scripts/validate.py`
- `src/scripts/benchmark.py`

## Validation Commands

Validate dataset and training config paths without starting training:

```bash
uv run --python 3.11 python src/scripts/train.py --validate-only --data-config configs/data.yaml --train-config configs/train.yaml
```

Run evaluation on the validation split:

```bash
uv run --python 3.11 python src/scripts/validate.py --data-config configs/data.yaml --train-config configs/train.yaml --split val
```

Run benchmarking on the test split:

```bash
uv run --python 3.11 python src/scripts/benchmark.py --data-config configs/data.yaml --train-config configs/train.yaml --split test
```

## Common Dataset Issues

### Dataset root not found

Cause:

- `path` points to a missing directory

### Train or validation directory not found

Cause:

- `train` or `val` does not resolve to an existing directory

### Invalid `names`

Cause:

- `names` is missing
- `names` is empty
- `names` contains blank or non-string values

### Model not found

Cause:

- `configs/train.yaml` points to a missing local model path

## Practical Checklist

Before training, confirm:

- dataset images are split into `train`, `val`, and optionally `test`
- each image has a matching `.txt` label file
- label class IDs match the order of `names`
- `configs/data.yaml` points to the correct dataset root
- `configs/train.yaml` points to a valid YOLO model
