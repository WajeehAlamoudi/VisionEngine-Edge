from __future__ import annotations

import logging
from pathlib import Path

from core.config import ModelConfig
from ...stable_id import StableIdMap
from ...types import InferenceResult

log = logging.getLogger(__name__)

# What NvDs writes into object_id when an object has no track — either because
# tracking is off, or because the tracker has not yet opened one for it.
_UNTRACKED = (1 << 64) - 1


class DeepStreamDetections:
    """
    Turns what nvinfer and nvtracker found into InferenceResult rows.

    This is the detection half of the DeepStream runtime: the class map, the
    confidence floor, the camera's class filter, box clamping, and the stable
    track_id. The other half — connecting to the camera, decoding, and moving
    buffers through the pipeline — is runtime.py.

    Deliberately not a Detector subclass. That ABC is defined by
    infer(frame) -> boxes, and here there is no frame: nvinfer runs inside the
    pipeline that decoded it and attaches its results to the buffer as
    metadata. Forcing this into that shape is what an earlier version did, and
    it cost the pipeline a CPU decode and a copy per frame to hand over a frame
    nothing needed.
    """

    def __init__(self, model: ModelConfig, classes: list[str]) -> None:
        self._model = model
        # None means "keep every class the model reports"; a list narrows it to
        # what this camera asked for.
        self._active = set(classes) if classes else None
        self._idx_to_name: dict[int, str] = {}
        self._ids = StableIdMap()

    @property
    def class_count(self) -> int:
        return len(self._idx_to_name)

    # ── class names ───────────────────────────────────────────────────────────

    def load_labels(self) -> None:
        """
        Read the class map from the label file nvinfer is configured with.

        models.yaml `classes` cannot stand in for it. That is a filter naming
        the classes wanted, in any order and any subset, whereas class_id
        indexes into the model's own ordered list. A ten-class model narrowed
        to two would map ids 0 and 1 to those two names and mislabel every
        other detection.
        """
        cfg_path = Path(self._model.ds_infer_config)
        value = None
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("labelfile-path") and "=" in line:
                value = line.partition("=")[2].strip()
                break

        if not value:
            raise RuntimeError(
                f"{cfg_path} has no labelfile-path — it is required, because "
                f"class_id indexes into the model's own ordered class list"
            )

        candidate = Path(value)
        label_path = candidate if candidate.is_absolute() else cfg_path.parent / candidate
        if not label_path.is_file():
            raise FileNotFoundError(
                f"labelfile-path '{value}' resolves to a missing {label_path}"
            )

        names = [n.strip() for n in
                 label_path.read_text(encoding="utf-8").splitlines() if n.strip()]
        if not names:
            raise RuntimeError(f"label file {label_path} is empty")

        # A camera class the model never emits would silently match nothing,
        # which looks identical to a camera that simply sees no people.
        if self._active is not None:
            unknown = sorted(self._active - set(names))
            if unknown:
                raise RuntimeError(
                    f"camera classes {unknown} are not in {label_path} "
                    f"(which has: {', '.join(names)}) — they could never match"
                )

        self._idx_to_name = dict(enumerate(names))
        log.info("deepstream: %d class names from %s", len(names), label_path)

    # ── metadata ──────────────────────────────────────────────────────────────

    def parse(self, pyds, sample, width: int, height: int) -> list[InferenceResult]:
        """
        Walk the metadata attached to one processed buffer.

        NvDs metadata is a C linked list rather than a Python container, hence
        the explicit .next traversal. The batch holds one frame here, but
        writing it as a real traversal means batching more cameras later needs
        no rewrite.
        """
        # hash() on a PyGObject yields the underlying C pointer — the pyds
        # idiom for handing a buffer to the metadata API.
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(sample.get_buffer()))
        if batch_meta is None:
            return []

        out: list[InferenceResult] = []
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                result = self._to_result(
                    pyds.NvDsObjectMeta.cast(l_obj.data), width, height)
                if result is not None:
                    out.append(result)
                l_obj = l_obj.next
            l_frame = l_frame.next
        return out

    def _to_result(self, obj, width: int, height: int) -> InferenceResult | None:
        """One object to one row, or None if it should be dropped."""
        name = self._idx_to_name.get(int(obj.class_id)) or str(obj.obj_label or "").strip()
        if not name:
            return None
        if self._active is not None and name not in self._active:
            return None

        confidence = float(obj.confidence)
        if confidence < self._model.confidence_threshold:
            return None

        # rect_params is already in nvstreammux's coordinate space, which the
        # runtime set to the source resolution — so no rescaling is needed.
        # Clamped because a tracker predicting through an occlusion can carry a
        # box past the frame edge, and an out-of-bounds box would corrupt the
        # zone tests downstream.
        rect = obj.rect_params
        x1 = max(0.0, float(rect.left))
        y1 = max(0.0, float(rect.top))
        x2 = min(float(width), x1 + float(rect.width))
        y2 = min(float(height), y1 + float(rect.height))
        if x2 <= x1 or y2 <= y1:
            return None

        raw_id = int(obj.object_id)
        return InferenceResult(
            class_name=name,
            confidence=confidence,
            bbox=[x1, y1, x2, y2],
            # The same StableIdMap the boxmot tracker uses, so a track_id
            # reaching the database has one shape whichever produced it.
            track_id=None if raw_id == _UNTRACKED else self._ids.get(raw_id),
        )
