from __future__ import annotations

import logging

import cv2
import numpy as np

from core.config import ModelConfig
from ..types import InferenceResult
from .base import Detector

log = logging.getLogger(__name__)


class HailoDetector(Detector):
    """
    Detection backend for a Hailo NPU (.hef) — runs entirely through
    HailoRT, never touches torch/Ultralytics. Quantization (INT8) is baked
    into the .hef at compile time by the Hailo Dataflow Compiler; there is
    no runtime precision flag to set here.

    hailo_platform (HailoRT's Python SDK) is only imported inside load()/
    infer(), never at module level — so importing this file, or building a
    detector registry that includes it, never requires the Hailo SDK/driver
    to be present. A camera running device: cuda or device: cpu is
    completely unaffected by this backend existing in the codebase, and a
    machine with no Hailo hardware can still import and run everything else
    normally.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self._device = None
        self._network_group = None
        self._network_group_params = None
        self._input_vstream_info = None
        self._output_vstream_infos = None
        self._input_vstream_params = None
        self._output_vstream_params = None
        self._input_shape: tuple[int, int, int] = (0, 0, 0)
        self._class_names: list[str] = []

    def load(self) -> None:
        # Imported lazily — hailo_platform only installs on Linux with the
        # Hailo driver present, and ships from Hailo's own SDK rather than
        # PyPI. This import only ever runs for a model whose config actually
        # says device: hailo (see registry.py's device→backend dispatch).
        from hailo_platform import (
            VDevice, HEF, ConfigureParams, HailoStreamInterface,
            InputVStreamParams, OutputVStreamParams, FormatType,
        )

        log.info("detector '%s': loading %s on hailo", self._cfg.id, self._cfg.path)

        hef = HEF(self._cfg.path)
        self._device = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe,
        )
        self._network_group = self._device.configure(hef, configure_params)[0]
        self._network_group_params = self._network_group.create_params()

        self._input_vstream_info = hef.get_input_vstream_infos()[0]
        self._output_vstream_infos = hef.get_output_vstream_infos()
        self._input_shape = self._input_vstream_info.shape  # (H, W, C)

        self._input_vstream_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.UINT8,
        )
        self._output_vstream_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32,
        )

        # A .hef carries no class names the way a .pt does — class order
        # must match the order the model was trained/exported with, which is
        # exactly the order given in this model's `classes:` list in
        # models.yaml. Getting that order wrong silently mislabels every
        # detection, so this is a hard requirement, not a convenience.
        self._class_names = list(self._cfg.classes)

        log.info(
            "detector '%s' ready — %d classes, input %s, device=hailo",
            self._cfg.id, len(self._class_names), self._input_shape,
        )

    def infer(self, frame, active_classes: list[str]) -> list[InferenceResult]:
        from hailo_platform import InferVStreams

        active = set(active_classes) if active_classes else None
        orig_h, orig_w = frame.shape[:2]
        model_input = self._preprocess(frame)

        with InferVStreams(
            self._network_group, self._input_vstream_params, self._output_vstream_params,
        ) as pipeline:
            with self._network_group.activate(self._network_group_params):
                input_data = {self._input_vstream_info.name: np.expand_dims(model_input, axis=0)}
                raw = pipeline.infer(input_data)

        return self._parse(raw, active, orig_size=(orig_w, orig_h))

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        target_h, target_w = self._input_shape[0], self._input_shape[1]
        resized = cv2.resize(frame, (target_w, target_h))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def _parse(
        self, raw: dict, active: set[str] | None, orig_size: tuple[int, int],
    ) -> list[InferenceResult]:
        """
        Decodes HailoRT output for a YOLO .hef compiled with Hailo's standard
        on-chip NMS postprocess — the default produced by Ultralytics' Hailo
        export path. Output is one array per class, each row
        [y_min, x_min, y_max, x_max, score] in 0..1 normalized coordinates.

        NOTE: this is Hailo's documented YOLO postprocess output shape, but
        has not been run against real hardware in this repo yet. Confirm
        against an actual inference call on the Jetson/Hailo module — if the
        compiled .hef doesn't have on-chip NMS baked in, this needs raw
        anchor/box decoding instead of this per-class-array parse.
        """
        out: list[InferenceResult] = []
        orig_w, orig_h = orig_size
        output_name = self._output_vstream_infos[0].name
        detections = raw[output_name][0]  # drop the batch dim

        for cls_idx, class_dets in enumerate(detections):
            if cls_idx >= len(self._class_names):
                continue
            class_name = self._class_names[cls_idx]
            if active is not None and class_name not in active:
                continue
            for det in class_dets:
                y_min, x_min, y_max, x_max, score = det[:5]
                if score < self._cfg.confidence_threshold:
                    continue
                out.append(InferenceResult(
                    class_name=class_name,
                    confidence=float(score),
                    bbox=[
                        float(x_min) * orig_w, float(y_min) * orig_h,
                        float(x_max) * orig_w, float(y_max) * orig_h,
                    ],
                    track_id=None,
                ))
        return out
