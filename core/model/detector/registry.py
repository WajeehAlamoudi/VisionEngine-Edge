from __future__ import annotations

from core.config import ModelConfig
from .base import Detector
from .ultralytics_detector import UltralyticsDetector

# Backend name → implementation. Add an entry here when a new detection
# backend is built (e.g. ONNX, Hailo); no change needed to ModelRunner or callers.
_BACKENDS: dict[str, type[Detector]] = {
    "ultralytics": UltralyticsDetector,
}


def build_detector(cfg: ModelConfig, backend: str = "ultralytics") -> Detector:
    cls = _BACKENDS.get(backend)
    if cls is None:
        raise RuntimeError(
            f"unknown detector backend '{backend}' — available: {', '.join(_BACKENDS)}"
        )
    return cls(cfg)
