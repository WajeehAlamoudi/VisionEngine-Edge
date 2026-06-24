from __future__ import annotations

import logging
from dataclasses import dataclass

from ultralytics import YOLO

from core.config import ModelConfig

log = logging.getLogger(__name__)


# ── raw inference result ──────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] absolute pixel coords


# ── single model runner ───────────────────────────────────────────────────────

class ModelRunner:
    """Wraps one YOLO model. Loaded once, reused across all frames for that model."""

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._model = None
        self._device = ""
        self._name_to_idx: dict[str, int] = {}

    def load(self) -> None:
        self._device = _resolve_device(self._cfg.device)
        log.info("model '%s': loading %s on %s", self._cfg.id, self._cfg.path, self._device)
        self._model = YOLO(self._cfg.path)
        # build name → class-index map from the model's own label list
        self._name_to_idx = {name: idx for idx, name in self._model.names.items()}
        log.info(
            "model '%s' ready — %d classes, device=%s",
            self._cfg.id, len(self._name_to_idx), self._device,
        )

    def predict(self, frame, active_classes: list[str]) -> list[InferenceResult]:
        """
        Run inference on a single frame.

        active_classes — the camera's effective class list (already resolved from config).
        Only detections whose class is in active_classes are returned.
        Passing the list to ultralytics as class indices lets it skip NMS
        for unwanted classes, which is faster than filtering after inference.
        """
        indices = [
            self._name_to_idx[c]
            for c in active_classes
            if c in self._name_to_idx
        ]

        # ultralytics imgsz expects (height, width); config stores [width, height]
        w, h = self._cfg.input_size
        imgsz = (h, w)

        results = self._model.predict(
            frame,
            conf=self._cfg.confidence_threshold,
            iou=self._cfg.iou_threshold,
            imgsz=imgsz,
            classes=indices or None,  # None = all classes
            device=self._device,
            verbose=False,
        )

        out: list[InferenceResult] = []
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                cls_name = self._model.names[cls_idx]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                out.append(InferenceResult(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                ))

        return out


# ── model registry ────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Loads and holds all model runners needed by enabled cameras.
    Each unique model_id is loaded exactly once — cameras sharing
    the same model_id reuse the same ModelRunner instance.
    """

    def __init__(self) -> None:
        self._runners: dict[str, ModelRunner] = {}

    def load_for_cameras(
            self,
            models: dict[str, ModelConfig],
            needed_ids: list[str],
    ) -> None:
        """Load only the models referenced by enabled cameras."""
        for model_id in dict.fromkeys(needed_ids):  # deduplicate, preserve order
            if model_id not in models:
                raise RuntimeError(
                    f"ModelRegistry: model_id '{model_id}' not found — "
                    f"this should have been caught by config validation"
                )
            runner = ModelRunner(models[model_id])
            runner.load()
            self._runners[model_id] = runner

    def get(self, model_id: str) -> ModelRunner:
        return self._runners[model_id]


# ── device resolution ─────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
        if torch.cuda.is_available():
            log.info("device auto-select: CUDA")
            return "cuda"
        if torch.backends.mps.is_available():
            log.info("device auto-select: MPS (Apple Metal)")
            return "mps"
    except ImportError:
        pass

    log.info("device auto-select: CPU")
    return "cpu"
