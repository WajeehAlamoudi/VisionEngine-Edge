from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml
from boxmot.trackers.bbox.botsort import BotSort

from core.config import ModelConfig
from ..types import InferenceResult
from .base import Tracker

log = logging.getLogger(__name__)

# Used when cfg.tracker doesn't point at a readable YAML file — keeps the
# tracker working even before a config file is deployed.
_DEFAULT_PARAMS = {
    "with_reid": False,
    "use_cmc": False,
}


class BoxMotTracker(Tracker):
    """
    BoT-SORT tracking via the boxmot library.

    Contains no detection model — update() only accepts detections a
    Detector already computed, and does Kalman-filter motion prediction +
    box matching (ReID appearance matching can be enabled later by passing
    reid_model to BotSort) to assign track_id. The detection model itself
    is free to be retrained/improved independently; this class never
    touches model weights.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self._tracker = None
        # local id<->name mapping, used only so boxmot's numeric cls column
        # can be translated back to a class name — stable across the life of
        # this tracker instance regardless of the detector's internal indices
        self._name_to_idx: dict[str, int] = {}
        self._idx_to_name: dict[int, str] = {}

    def load(self) -> None:
        self._name_to_idx = {name: i for i, name in enumerate(self._cfg.classes)}
        self._idx_to_name = {i: name for name, i in self._name_to_idx.items()}

        params = self._load_params()
        self._tracker = BotSort(**params)
        log.info(
            "tracker '%s' ready — boxmot BotSort (with_reid=%s, use_cmc=%s)",
            self._cfg.id, params.get("with_reid"), params.get("use_cmc"),
        )

    def _load_params(self) -> dict:
        path = Path(self._cfg.tracker)
        if not path.is_file():
            log.warning(
                "tracker '%s': config file '%s' not found — using defaults %s",
                self._cfg.id, path, _DEFAULT_PARAMS,
            )
            return dict(_DEFAULT_PARAMS)

        with path.open(encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        log.info("tracker '%s': loaded params from %s", self._cfg.id, path)
        return params

    def update(self, frame, detections: list[InferenceResult]) -> list[InferenceResult]:
        if not detections:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            dets = np.array([
                [*d.bbox, d.confidence, self._name_to_idx.get(d.class_name, -1)]
                for d in detections
            ], dtype=np.float32)

        tracks = self._tracker.update(dets, frame)

        out: list[InferenceResult] = []
        for xyxy, track_id, conf, cls_idx in zip(tracks.xyxy, tracks.id, tracks.conf, tracks.cls):
            out.append(InferenceResult(
                class_name=self._idx_to_name.get(int(cls_idx), "unknown"),
                confidence=float(conf),
                bbox=xyxy.tolist(),
                track_id=int(track_id),
            ))
        return out
