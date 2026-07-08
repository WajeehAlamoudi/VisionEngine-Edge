from __future__ import annotations

import logging

from ultralytics import YOLO

from core.config import ModelConfig
from .device import _resolve_device
from .types import InferenceResult

log = logging.getLogger(__name__)


class ModelRunner:
    """
    Wraps one YOLO model instance.

    use_tracker=False → model.predict() — stateless, safe to share between cameras
    use_tracker=True  → model.track()  — stateful, one dedicated instance per camera
    """

    def __init__(self, cfg: ModelConfig, use_tracker: bool = False, tracker: str = "botsort.yaml") -> None:
        self._cfg         = cfg
        self._use_tracker = use_tracker
        self._tracker     = tracker
        self._model       = None
        self._device      = ""
        self._name_to_idx: dict[str, int] = {}

    def load(self) -> None:
        self._device = _resolve_device(self._cfg.device)
        log.info("model '%s': loading %s on %s (tracker=%s)", self._cfg.id, self._cfg.path, self._device, self._use_tracker)
        self._model = YOLO(self._cfg.path)
        self._name_to_idx = {name: idx for idx, name in self._model.names.items()}
        log.info(
            "model '%s' ready — %d classes, device=%s",
            self._cfg.id, len(self._name_to_idx), self._device,
        )

    def run(self, frame, active_classes: list[str]) -> list[InferenceResult]:
        if self._use_tracker:
            return self._track(frame, active_classes)
        return self._predict(frame, active_classes)

    def _predict(self, frame, active_classes: list[str]) -> list[InferenceResult]:
        indices = self._class_indices(active_classes)
        w, h = self._cfg.input_size
        results = self._model.predict(
            frame,
            conf=self._cfg.confidence_threshold,
            iou=self._cfg.iou_threshold,
            imgsz=(h, w),
            classes=indices or None,
            device=self._device,
            verbose=False,
        )
        return self._parse(results)

    def _track(self, frame, active_classes: list[str]) -> list[InferenceResult]:
        indices = self._class_indices(active_classes)
        w, h = self._cfg.input_size
        results = self._model.track(
            frame,
            conf=self._cfg.confidence_threshold,
            iou=self._cfg.iou_threshold,
            imgsz=(h, w),
            classes=indices or None,
            device=self._device,
            tracker=self._tracker,
            persist=True,       # keep tracker state across frames
            verbose=False,
        )
        return self._parse(results)

    def _class_indices(self, active_classes: list[str]) -> list[int]:
        return [self._name_to_idx[c] for c in active_classes if c in self._name_to_idx]

    def _parse(self, results) -> list[InferenceResult]:
        out: list[InferenceResult] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_idx  = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else None
                out.append(InferenceResult(
                    class_name=self._model.names[cls_idx],
                    confidence=float(box.conf[0]),
                    bbox=box.xyxy[0].tolist(),
                    track_id=track_id,
                ))
        return out
