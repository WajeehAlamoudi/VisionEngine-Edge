from __future__ import annotations

from core.config import ModelConfig
from .base import Detector
from .ultralytics_detector import UltralyticsDetector
from .deepstream_detector import DeepStreamDetector

# models.yaml `runtime:` value → implementation. The runtime IS the backend
# name, so there is no translation table: a second dictionary used to exist
# only because one `device` key was naming both the hardware and the code path.
_BACKENDS: dict[str, type[Detector]] = {
    "ultralytics": UltralyticsDetector,
    "deepstream": DeepStreamDetector,
}


def _backend_class(cfg: ModelConfig) -> type[Detector] | None:
    """
    The class that would run this model, without constructing or loading it.

    Both questions below are answered from class attributes so ModelRegistry
    and ModelRunner can decide before any SDK is touched — building a detector
    to ask would mean loading DeepStream just to find out whether the model
    needs its own instance.
    """
    return _BACKENDS.get(cfg.runtime)


def tracks_internally(cfg: ModelConfig) -> bool:
    """
    Whether this model assigns track_id inside the detector, meaning no
    separate Tracker should be built.

    Depends on use_tracker as well as the runtime: a backend capable of
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
    """Build the detector for a model config, from cfg.runtime unless a
    backend is explicitly named.

    Each backend module only imports its own SDK lazily, inside its own
    load()/infer() — never at module top level. That's what keeps switching
    one camera's model to a different runtime from ever touching, importing,
    or failing because of another runtime's SDK. A machine with no DeepStream
    install can still run every other backend normally, and vice versa.
    """
    cls = _BACKENDS.get(backend or cfg.runtime)
    if cls is None:
        raise RuntimeError(
            f"unknown runtime '{backend or cfg.runtime}' — "
            f"available: {', '.join(_BACKENDS)}"
        )
    return cls(cfg)
