from __future__ import annotations

from core.config import ModelConfig
from .base import Detector
from .ultralytics_detector import UltralyticsDetector
from .hailo_detector import HailoDetector

# Backend name → implementation. Add an entry here when a new detection
# backend is built; no change needed to ModelRunner or callers.
_BACKENDS: dict[str, type[Detector]] = {
    "ultralytics": UltralyticsDetector,
    "hailo": HailoDetector,
}

# Device (models.yaml `device:` value) → the backend that runs it. Every
# device here is one Ultralytics/torch itself can hand off to via its own
# `device=` argument — "auto" is resolved to one of cpu/cuda/mps first (see
# core/model/device.py), the other three are passed straight through.
# Hailo is the only device that isn't a torch device at all — it never goes
# through Ultralytics, it runs entirely on HailoRT.
_DEVICE_BACKENDS: dict[str, str] = {
    "auto":  "ultralytics",
    "cpu":   "ultralytics",
    "cuda":  "ultralytics",
    "mps":   "ultralytics",
    "hailo": "hailo",
}


def build_detector(cfg: ModelConfig, backend: str | None = None) -> Detector:
    """Build the detector for a model config. Backend is looked up from
    cfg.device via _DEVICE_BACKENDS, unless explicitly overridden.

    Each backend module only imports its own runtime SDK lazily, inside its
    own load()/infer() — never at module top level. That's what keeps
    switching one camera's model to a different device (cpu/cuda/hailo/...)
    from ever touching, importing, or failing because of another device's
    SDK. A machine with no Hailo hardware/driver can still run every other
    backend normally, and vice versa.
    """
    resolved = backend or _DEVICE_BACKENDS.get(cfg.device)
    if resolved is None:
        raise RuntimeError(
            f"unknown device '{cfg.device}' — no backend mapped for it. "
            f"Supported devices: {', '.join(_DEVICE_BACKENDS)}"
        )
    cls = _BACKENDS.get(resolved)
    if cls is None:
        raise RuntimeError(
            f"unknown detector backend '{resolved}' — available: {', '.join(_BACKENDS)}"
        )
    return cls(cfg)
