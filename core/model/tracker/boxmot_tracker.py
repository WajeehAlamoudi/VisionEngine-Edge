from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import yaml

from core.config import ModelConfig
from ..accelerator import resolve_accelerator
from ..stable_id import StableIdMap
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
# accelerator from models.yaml is NOT usable directly: torch.device() raises
# on "coreml", so a CoreML detector with with_reid: true used to crash at
# load. They are genuinely separate devices - one runs the detector, torch
# runs ReID.
_TORCH_DEVICES = ("cpu", "cuda", "mps")

# Tracker algorithms supported by the Ultralytics/BoxMot path.  The detector
# remains unchanged; each class only associates detections across frames.
_TRACKER_ALGORITHMS = ("botsort", "bytetrack")

# BoxMot tracker instances are per camera, but TensorRT ReID calls ultimately
# share one CUDA context in this process. Serialize only the tracker/ReID call;
# capture and detector inference remain independent per camera.
_TENSORRT_REID_LOCK = threading.Lock()


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

        class SingleLoadTensorRTBackend(tensorrt_backend.TensorRTBackend):
            """Avoid BoxMot loading the same TensorRT engine twice per tracker."""

            def __init__(self, *args, **kwargs):
                # BaseModelBackend.__init__ dynamically invokes load_model(),
                # then TensorRTBackend.__init__ invokes it again. The first call
                # creates a complete, discarded engine/context. Skip only that
                # base call; the TensorRT backend's own call still does the load.
                self._skip_base_load = True
                super().__init__(*args, **kwargs)

            def load_model(self, weights):
                if self._skip_base_load:
                    self._skip_base_load = False
                    return
                return super().load_model(weights)

        return SingleLoadTensorRTBackend
    raise RuntimeError(f"unsupported reid_backend '{name}'")


def _import_tracker_algorithm(name: str):
    """Import only the BoxMot tracker selected by the deployed YAML file."""
    if name == "botsort":
        from boxmot.trackers.bbox.botsort import BotSort
        return BotSort
    if name == "bytetrack":
        from boxmot.trackers.bbox.bytetrack import ByteTrack
        return ByteTrack
    raise RuntimeError(
        f"unsupported tracker algorithm '{name}' — expected one of "
        f"{', '.join(_TRACKER_ALGORITHMS)}"
    )


class BoxMotTracker(Tracker):
    """
    Selectable BoT-SORT or ByteTrack association via the boxmot library.

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
        # boxmot's raw integer track id → our stable UUID, for this tracker's
        # lifetime. Shared with the tracking detector backends so every
        # track_id written to the detections table has the same shape.
        self._ids = StableIdMap()
        self._uses_tensorrt_reid = False

    def load(self) -> None:
        # Imported here, not at module level, for the reason _import_reid_backend
        # gives below: a device that runs another tracker entirely - DeepStream's
        # nvtracker - has no boxmot and no torch installed, and a top-level
        # import made core.model unimportable there. That took the debug tool and
        # anything else touching the model layer down with it, on a device where
        # this class was never going to be built.
        self._name_to_idx = {name: i for i, name in enumerate(self._cfg.classes)}
        self._idx_to_name = {i: name for name, i in self._name_to_idx.items()}

        params = self._load_params()
        algorithm = str(params.pop("algorithm", "botsort")).strip().lower()
        tracker_cls = _import_tracker_algorithm(algorithm)

        # Popped, not passed through: BotSort wants a constructed reid_model,
        # not the pieces it is built from.
        reid_weights = params.pop("reid_weights", _DEFAULT_REID_WEIGHTS)
        reid_backend = params.pop("reid_backend", "pytorch")
        reid_device = params.pop("reid_device", "auto")
        reid_half = params.pop("reid_half", False)

        if algorithm == "botsort" and params.get("with_reid"):
            params["reid_model"] = self._build_reid(
                reid_weights, reid_backend, reid_device, reid_half
            )
            self._uses_tensorrt_reid = reid_backend == "tensorrt"

        self._tracker = tracker_cls(**params)
        log.info(
            "tracker '%s' ready — boxmot %s (with_reid=%s)",
            self._cfg.id, algorithm, bool(params.get("with_reid")),
        )

    def _reid_device(self, requested: str) -> torch.device:
        """
        Resolve the torch device the ReID network runs on.

        "auto" follows the detector when the detector is on a torch device, and
        falls back to cpu when it is not - a CoreML detector still needs its
        ReID on cpu or cuda.
        """
        import torch      # boxmot's dependency, not ours — see load()

        if requested == "auto":
            detector_accelerator = resolve_accelerator(self._cfg.accelerator)
            if detector_accelerator in _TORCH_DEVICES:
                return torch.device(detector_accelerator)
            log.warning(
                "tracker '%s': accelerator '%s' is not a torch device - running "
                "ReID on cpu. Set reid_device explicitly to override.",
                self._cfg.id, detector_accelerator,
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
                "tracker '%s': reid_half ignored - no benefit on reid device %s",
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

        if self._uses_tensorrt_reid:
            with _TENSORRT_REID_LOCK:
                tracks = self._tracker.update(dets, frame)
        else:
            tracks = self._tracker.update(dets, frame)

        out: list[InferenceResult] = []
        for xyxy, track_id, conf, cls_idx in zip(tracks.xyxy, tracks.id, tracks.conf, tracks.cls):
            out.append(InferenceResult(
                class_name=self._idx_to_name.get(int(cls_idx), "unknown"),
                confidence=float(conf),
                bbox=xyxy.tolist(),
                track_id=self._ids.get(int(track_id)),
            ))
        return out
