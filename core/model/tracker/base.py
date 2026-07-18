from __future__ import annotations

from abc import ABC, abstractmethod

from core.config import ModelConfig
from ..types import InferenceResult


class Tracker(ABC):
    """
    One dedicated instance per camera — trackers are stateful (they remember
    prior frames to assign consistent track_ids), so unlike Detector they can
    never be shared across cameras.

    update() consumes detections already produced by a Detector — a Tracker
    implementation must contain no detection model of its own. Its only job
    is deciding which of this frame's boxes are the same object as a box
    from a previous frame (motion prediction + matching, optionally
    appearance/ReID matching), and assigning track_id accordingly.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def update(self, frame, detections: list[InferenceResult]) -> list[InferenceResult]: ...
