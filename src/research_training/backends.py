"""Explicit backend policy, recorded per run without changing legacy settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch


def configure_backends(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Apply opt-in cuDNN tuning; absent config preserves legacy default policy."""
    settings = config.get("runtime", {}).get("cudnn", {})
    if not isinstance(settings, dict):
        raise ValueError("runtime.cudnn must be a mapping")
    flags = {
        "enabled": settings.get("enabled", True),
        "benchmark": settings.get("benchmark", False),
        "deterministic": settings.get("deterministic", False),
    }
    if any(not isinstance(value, bool) for value in flags.values()):
        raise ValueError("cuDNN enabled/benchmark/deterministic settings must be booleans")
    if flags["benchmark"] and (not flags["enabled"] or flags["deterministic"]):
        raise ValueError("cuDNN benchmark requires enabled=true and deterministic=false")
    available = cast(Callable[[], bool], torch.backends.cudnn.is_available)()
    if device.type == "cuda" and flags["enabled"] and not available:
        raise RuntimeError("cuDNN requested but unavailable in this PyTorch installation")
    torch.backends.cudnn.enabled = flags["enabled"]
    torch.backends.cudnn.benchmark = flags["benchmark"]
    torch.backends.cudnn.deterministic = flags["deterministic"]
    return {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": cast(Callable[[], int | None], torch.backends.cudnn.version)(),
        "cudnn_available": available,
        "cudnn_enabled": flags["enabled"],
        "cudnn_active": device.type == "cuda" and available and flags["enabled"],
        "cudnn_benchmark": flags["benchmark"],
        "cudnn_deterministic": flags["deterministic"],
    }
