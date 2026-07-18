from __future__ import annotations

import logging

from core.config import ModelConfig
from .detector import Detector, build_detector
from .tracker import Tracker, build_tracker
from .types import InferenceResult

log = logging.getLogger(__name__)


class ModelRunner:
    """
    Wraps one camera's detection (+ optional tracking) pipeline.

    Detection always runs through Detector — exactly once per frame, whether
    or not tracking is enabled. When use_tracker=True, those detections are
    handed to a Tracker, which assigns track_id (motion prediction + box
    matching) and contains no detection model of its own. This keeps the
    detection model (retrained/improved independently over time) fully
    decoupled from the tracking algorithm.
    """

    def __init__(self, cfg: ModelConfig, use_tracker: bool = False, tracker: str = "botsort.yaml") -> None:
        self._cfg         = cfg
        self._use_tracker = use_tracker
        self._tracker_algorithm = tracker   # kept for back-compat/logging; unused by the boxmot backend
        self._detector: Detector | None = None
        self._tracker:  Tracker  | None = None

    def load(self) -> None:
        self._detector = build_detector(self._cfg)
        self._detector.load()
        if self._use_tracker:
            self._tracker = build_tracker(self._cfg)
            self._tracker.load()

    def run(self, frame, active_classes: list[str]) -> list[InferenceResult]:
        detections = self._detector.infer(frame, active_classes)
        if self._tracker is None:
            return detections
        return self._tracker.update(frame, detections)
