"""Validate GPU availability for Sprint 1."""

from __future__ import annotations

import json
import sys
from typing import Any

import torch
from loguru import logger


def collect_gpu_status() -> dict[str, Any]:
    """Collect the current CUDA and GPU status."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

    return {
        "success": cuda_available,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": gpu_name,
    }


def main() -> int:
    """Run the GPU validation script."""
    status = collect_gpu_status()

    if status["cuda_available"]:
        logger.info("CUDA is available on {}", status["gpu_name"])
    else:
        logger.error("CUDA is not available")

    print(json.dumps(status, indent=2))
    return 0 if status["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
