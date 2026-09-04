from __future__ import annotations

import logging

from ultralytics import YOLO

from core.config import ModelConfig
from ..device import _resolve_device
from ..types import InferenceResult
from .base import Detector

log = logging.getLogger(__name__)


class UltralyticsDetector(Detector):
    """Detection-only wrapper around one Ultralytics YOLO model (model.predict())."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self._model = None
        self._device = ""
        self._half = False
        self._name_to_idx: dict[str, int] = {}

    def load(self) -> None:
        self._device = _resolve_device(self._cfg.accelerator)
        # half only helps on a GPU — on cpu/mps it typically gives no speedup
        # (sometimes slower), so a stray half: true in config is silently a
        # no-op there instead of doing something counterproductive.
        self._half = self._cfg.half and self._device == "cuda"
        if self._cfg.half and not self._half:
            log.info(
                "detector '%s': half=true ignored — no benefit on device=%s",
                self._cfg.id, self._device,
            )
        log.info("detector '%s': loading %s on %s (half=%s)", self._cfg.id, self._cfg.path, self._device, self._half)
        self._model = YOLO(self._cfg.path)
        self._name_to_idx = {name: idx for idx, name in self._model.names.items()}
        log.info(
            "detector '%s' ready — %d classes, device=%s",
            self._cfg.id, len(self._name_to_idx), self._device,
        )

    def infer(self, frame, active_classes: list[str]) -> list[InferenceResult]:
        indices = self._class_indices(active_classes)
        w, h = self._cfg.input_size
        predict_kwargs = dict(
            conf=self._cfg.confidence_threshold,
            iou=self._cfg.iou_threshold,
            imgsz=(h, w),
            classes=indices or None,
            verbose=False,
        )
        # A .mlpackage's execution target (CPU/GPU/Neural Engine) is fixed
        # at export time — Ultralytics' CoreML backend doesn't accept a
        # device=/half= override the way its torch backends do.
        if self._cfg.accelerator != "coreml":
            predict_kwargs["device"] = self._device
            predict_kwargs["half"] = self._half
        results = self._model.predict(frame, **predict_kwargs)
        return self._parse(results)

    def _class_indices(self, active_classes: list[str]) -> list[int]:
        return [self._name_to_idx[c] for c in active_classes if c in self._name_to_idx]

    def _parse(self, results) -> list[InferenceResult]:
        out: list[InferenceResult] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                out.append(InferenceResult(
                    class_name=self._model.names[cls_idx],
                    confidence=float(box.conf[0]),
                    bbox=box.xyxy[0].tolist(),
                    track_id=None,
                ))
        return out
