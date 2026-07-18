from __future__ import annotations

from core.config import ModelConfig
from .base import Tracker
from .boxmot_tracker import BoxMotTracker

# Backend name → implementation. Every backend registered here must consume
# detections a Detector already computed — no backend may contain its own
# detection model. Add an entry here when a new tracking algorithm is added
# (e.g. a different boxmot tracker, or a ReID-enabled variant).
_BACKENDS: dict[str, type[Tracker]] = {
    "boxmot": BoxMotTracker,
}


def build_tracker(cfg: ModelConfig, backend: str = "boxmot") -> Tracker:
    cls = _BACKENDS.get(backend)
    if cls is None:
        raise RuntimeError(
            f"unknown tracker backend '{backend}' — available: {', '.join(_BACKENDS)}"
        )
    return cls(cfg)
