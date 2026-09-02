from __future__ import annotations

from core.config import ModelConfig
from .base import Detector
from .ultralytics_detector import UltralyticsDetector
from .deepstream_detector import DeepStreamDetector

# Backend name → implementation. Add an entry here when a new detection
# backend is built; no change needed to ModelRunner or callers.
_BACKENDS: dict[str, type[Detector]] = {
    "ultralytics": UltralyticsDetector,
    "deepstream": DeepStreamDetector,
}

# Device (models.yaml `device:` value) → the backend that runs it. auto/cpu/
# cuda/mps are torch devices Ultralytics hands off to via its own `device=`
# argument ("auto" is resolved to one of cpu/cuda/mps first, see
# core/model/device.py). coreml also runs through Ultralytics, but a
# .mlpackage's execution target is fixed at export time, not by `device=`
# (see UltralyticsDetector.infer).
#
# deepstream is an NVIDIA GPU too, but it is listed as its own device because
# it selects a different runtime for the same hardware: nvinfer inside a
# GStreamer pipeline rather than Ultralytics, and tracking fused into that same
# pipeline. cuda continues to mean Ultralytics, unchanged.
_DEVICE_BACKENDS: dict[str, str] = {
    "auto":       "ultralytics",
    "cpu":        "ultralytics",
    "cuda":       "ultralytics",
    "mps":        "ultralytics",
    "coreml":     "ultralytics",
    "deepstream": "deepstream",
}


def _backend_class(cfg: ModelConfig) -> type[Detector] | None:
    """
    The class that would run this model, without constructing or loading it.

    Both questions below are answered from class attributes so ModelRegistry
    and ModelRunner can decide before any SDK is touched — building a detector
    to ask would mean loading DeepStream just to find out whether the model
    needs its own instance.
    """
    backend = _DEVICE_BACKENDS.get(cfg.device)
    return _BACKENDS.get(backend) if backend else None


def tracks_internally(cfg: ModelConfig) -> bool:
    """
    Whether this model assigns track_id inside the detector, meaning no
    separate Tracker should be built.

    Depends on use_tracker as well as the backend: a backend capable of
    tracking internally only does so when tracking is asked for. DeepStream
    with use_tracker false builds no nvtracker at all, and then behaves like
    any other detection-only backend.
    """
    cls = _backend_class(cfg)
    return bool(cls and cls.tracks_internally and cfg.use_tracker)


def is_shareable(cfg: ModelConfig) -> bool:
    """Whether one instance of this model's backend can serve several cameras."""
    cls = _backend_class(cfg)
    return cls is None or cls.shareable


def build_detector(cfg: ModelConfig, backend: str | None = None) -> Detector:
    """Build the detector for a model config. Backend is looked up from
    cfg.device via _DEVICE_BACKENDS, unless explicitly overridden.

    Each backend module only imports its own runtime SDK lazily, inside its
    own load()/infer() — never at module top level. That's what keeps
    switching one camera's model to a different device (cpu/cuda/deepstream/...)
    from ever touching, importing, or failing because of another device's
    SDK. A machine with no DeepStream install can still run every other
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
