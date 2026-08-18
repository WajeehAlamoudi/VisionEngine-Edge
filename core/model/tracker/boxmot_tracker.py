from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml
from boxmot.trackers.bbox.botsort import BotSort

from core.config import ModelConfig
from ..device import _resolve_device
from ..types import InferenceResult
from .base import Tracker

log = logging.getLogger(__name__)

# Used when cfg.tracker doesn't point at a readable YAML file — keeps the
# tracker working even before a config file is deployed.
_DEFAULT_PARAMS = {
    "with_reid": False,
    "use_cmc": False,
}

# Small, widely-used person-ReID checkpoint — auto-downloaded by boxmot on
# first use, same pattern as Ultralytics auto-downloading YOLO weights.
_DEFAULT_REID_WEIGHTS = "osnet_x0_25_msmt17.pt"

# ReID keys consumed by this class rather than passed through to BotSort.
# BotSort takes a constructed reid_model object, not weights/device/precision,
# so these are popped out of the params dict before it is expanded.
#
# boxmot ships six ReID backends, but only these two are safe to select here.
# Each of the others declares a pip requirement that boxmot auto-installs when
# unsatisfied - onnxruntime==1.24.3, openvino>=2025.2.0, ai-edge-litert - and
# that installer is destructive on a device whose CUDA stack comes from the OS
# rather than pip: it will pull a generic torch wheel and leave the GPU
# unusable. pytorch needs nothing extra, and the tensorrt path disables the
# installer explicitly because TensorRT is a system library here.
_REID_BACKENDS = ("pytorch", "tensorrt")

# Named only to give a useful error rather than "unknown backend".
_REID_BACKENDS_REQUIRING_INSTALL = ("onnx", "openvino", "tflite", "torchscript")

# The ReID network is a torch model, so it needs a torch device. The detector's
# device from models.yaml is NOT usable: torch.device() raises on "hailo" and
# "coreml", so a Hailo detector with with_reid: true used to crash at load.
# They are genuinely separate devices — Hailo runs the detector, torch runs ReID.
_TORCH_DEVICES = ("cpu", "cuda", "mps")


def _skip_dependency_install(*_args, **_kwargs) -> tuple:
    """Stand-in for boxmot's ReID dependency auto-installer. Installs nothing."""
    return ()


def _import_reid_backend(name: str):
    """
    Import a boxmot ReID backend on demand.

    Kept lazy so a device without TensorRT installed can still run the pytorch
    backend - a module-level import would make the whole tracker unimportable
    there.
    """
    if name == "pytorch":
        from boxmot.reid.backends.pytorch_backend import PyTorchBackend
        return PyTorchBackend
    if name == "tensorrt":
        from boxmot.reid.backends import tensorrt_backend

        # boxmot's TensorRT backend calls an auto-installer on every load_model,
        # looking for a pip package named "nvidia-tensorrt". On platforms where
        # TensorRT ships with the OS - Jetson via JetPack - that package does
        # not exist and cannot be built, and the attempted install replaces the
        # vendor torch build with a generic wheel whose bundled CUDA runtime the
        # driver cannot use. That leaves the device with CUDA unavailable.
        #
        # TensorRT is a system library here, exactly as it is for the detector
        # backends, so the check is disabled and the import is trusted. If
        # tensorrt is genuinely missing, the ImportError inside load_model is
        # the correct failure - a clear message rather than a broken install.
        tensorrt_backend.ensure_reid_backend_requirements = _skip_dependency_install
        return tensorrt_backend.TensorRTBackend
    raise RuntimeError(f"unsupported reid_backend '{name}'")


class BoxMotTracker(Tracker):
    """
    BoT-SORT tracking via the boxmot library.

    Contains no detection model — update() only accepts detections a
    Detector already computed, and does Kalman-filter motion prediction +
    box matching (ReID appearance matching can be enabled later by passing
    reid_model to BotSort) to assign track_id. The detection model itself
    is free to be retrained/improved independently; this class never
    touches model weights.

    track_id exposed to the rest of the system is a UUID, not boxmot's raw
    integer. boxmot's own counter resets to 1 on every process restart —
    left as-is internally since it's core to the tracking algorithm — but
    reusing that integer directly as our persisted track_id would let a
    track from today collide with an unrelated track after a restart next
    week. Mapping each raw integer to a freshly-generated UUID the first
    time it's seen removes that collision risk entirely: a new tracker
    instance (created on every restart) starts with an empty map, so IDs
    from a previous run can never resurface.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self._tracker = None
        # local id<->name mapping, used only so boxmot's numeric cls column
        # can be translated back to a class name — stable across the life of
        # this tracker instance regardless of the detector's internal indices
        self._name_to_idx: dict[str, int] = {}
        self._idx_to_name: dict[int, str] = {}
        # boxmot's raw integer track id → our stable UUID, for this tracker's lifetime
        self._id_map: dict[int, str] = {}

    def load(self) -> None:
        self._name_to_idx = {name: i for i, name in enumerate(self._cfg.classes)}
        self._idx_to_name = {i: name for name, i in self._name_to_idx.items()}

        params = self._load_params()

        # Popped, not passed through: BotSort wants a constructed reid_model,
        # not the pieces it is built from.
        reid_weights = params.pop("reid_weights", _DEFAULT_REID_WEIGHTS)
        reid_backend = params.pop("reid_backend", "pytorch")
        reid_device  = params.pop("reid_device", "auto")
        reid_half    = params.pop("reid_half", False)

        if params.get("with_reid"):
            params["reid_model"] = self._build_reid(
                reid_weights, reid_backend, reid_device, reid_half
            )

        self._tracker = BotSort(**params)
        log.info(
            "tracker '%s' ready — boxmot BotSort (with_reid=%s, use_cmc=%s)",
            self._cfg.id, params.get("with_reid"), params.get("use_cmc"),
        )

    def _reid_device(self, requested: str) -> torch.device:
        """
        Resolve the torch device the ReID network runs on.

        "auto" follows the detector when the detector is on a torch device, and
        falls back to cpu when it is not - a Hailo or CoreML detector still
        needs its ReID on cpu or cuda.
        """
        if requested == "auto":
            detector_device = _resolve_device(self._cfg.device)
            if detector_device in _TORCH_DEVICES:
                return torch.device(detector_device)
            log.warning(
                "tracker '%s': detector device '%s' is not a torch device - "
                "running ReID on cpu. Set reid_device explicitly to override.",
                self._cfg.id, detector_device,
            )
            return torch.device("cpu")

        if requested not in _TORCH_DEVICES:
            raise RuntimeError(
                f"tracker '{self._cfg.id}': reid_device '{requested}' is not a "
                f"torch device - expected auto, {', '.join(_TORCH_DEVICES)}"
            )
        return torch.device(requested)

    def _build_reid(self, weights: str, backend: str, requested_device: str,
                    half: bool):
        """
        Construct the ReID backend.

        Backend modules are imported lazily so a device without TensorRT (or
        OpenVINO, or tflite) can still run the pytorch backend - the same rule
        the detector backends follow for their SDKs.
        """
        if backend in _REID_BACKENDS_REQUIRING_INSTALL:
            raise RuntimeError(
                f"tracker '{self._cfg.id}': reid_backend '{backend}' is not "
                f"supported - boxmot would pip install its runtime on first "
                f"load, which replaces a vendor torch build on devices whose "
                f"CUDA stack comes from the OS. Use one of {', '.join(_REID_BACKENDS)}."
            )
        if backend not in _REID_BACKENDS:
            raise RuntimeError(
                f"tracker '{self._cfg.id}': unknown reid_backend '{backend}' - "
                f"expected one of {', '.join(_REID_BACKENDS)}"
            )

        device = self._reid_device(requested_device)

        # half only helps on a real GPU; on cpu/mps it gives no speedup and can
        # be slower, so a stray reid_half: true is a logged no-op there rather
        # than something counterproductive.
        use_half = half and device.type == "cuda"
        if half and not use_half:
            log.info(
                "tracker '%s': reid_half ignored - no benefit on device=%s",
                self._cfg.id, device,
            )

        cls = _import_reid_backend(backend)
        model = cls(weights, device, half=use_half)
        log.info(
            "tracker '%s': ReID ready — %s backend on %s (half=%s, weights=%s)",
            self._cfg.id, backend, device, use_half, weights,
        )
        return model

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
                track_id=self._stable_id(int(track_id)),
            ))
        return out

    def _stable_id(self, raw_id: int) -> str:
        if raw_id not in self._id_map:
            self._id_map[raw_id] = str(uuid.uuid4())
        return self._id_map[raw_id]
