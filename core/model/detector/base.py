from __future__ import annotations

from abc import ABC, abstractmethod

from core.config import ModelConfig
from ..types import InferenceResult


class Detector(ABC):
    """
    Stateless, detection-only backend. Safe to share between cameras that use
    the same model_id — carries no per-camera state (unlike Tracker).

    infer() takes a raw frame and the camera's active class list, and returns
    detections with track_id always None — tracking is a separate concern,
    added by the tracker/ layer when a camera has use_tracker=True.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def infer(self, frame, active_classes: list[str]) -> list[InferenceResult]: ...
