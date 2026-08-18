from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .strict import ID_PATTERN, Reader, describe

_FILE = "models.yaml"

_MODEL_KEYS = (
    "id", "name", "version", "path", "device", "classes",
    "confidence_threshold", "iou_threshold", "input_size",
    "use_tracker", "tracker", "half",
)

# Must stay in sync with _DEVICE_BACKENDS in core/model/detector/registry.py.
# Duplicated rather than imported because that module pulls in ultralytics and
# torch at import time, and config validation must stay free of heavy deps.
DEVICES = ("auto", "cpu", "cuda", "mps", "coreml", "hailo")

# Which devices each weight format can actually run on. The sample has always
# documented this; nothing enforced it until now.
#
# .engine deliberately excludes "auto": a TensorRT engine cannot load on CPU,
# and auto resolves to CPU whenever CUDA is unavailable, turning a missing GPU
# into an obscure runtime failure instead of a clear config error.
_FORMAT_DEVICES: dict[str, tuple[str, ...]] = {
    ".pt":        ("auto", "cpu", "cuda", "mps"),
    ".onnx":      ("auto", "cpu", "cuda", "mps"),
    ".engine":    ("cuda",),
    ".hef":       ("hailo",),
    ".mlpackage": ("coreml",),
    ".tflite":    ("cpu",),
}

# YOLO models use a stride of 32, so other sizes get resized internally.
_STRIDE = 32


@dataclass
class ModelConfig:
    id: str
    name: str
    version: str
    path: str
    device: str             # auto | cpu | cuda | mps | coreml | hailo
    classes: list[str]      # what this deployment expects the model to detect
    confidence_threshold: float
    iou_threshold: float
    input_size: list[int]   # [width, height]
    use_tracker: bool        # true = BoT-SORT tracker ON → track_id populated per object
    tracker: str             # path to the tracker params file, e.g. config/botsort_tracker.yaml
    half: bool               # FP16 inference — only applied when device resolves to cuda;
                              # ignored on cpu/mps, where it gives no benefit (see device.py)


def _check_format(r: Reader, path_value: str, device: str) -> None:
    """The weight file's extension must match the device that will run it."""
    if not path_value or not device:
        return

    suffix = Path(path_value).suffix.lower()
    allowed = _FORMAT_DEVICES.get(suffix)

    if allowed is None:
        r.error(
            r.path_of("path"),
            f"unrecognised model format '{suffix}' - "
            f"expected one of {', '.join(sorted(_FORMAT_DEVICES))}",
        )
        return

    if device not in allowed:
        r.error(
            r.path_of("device"),
            f"device '{device}' cannot run a {suffix} model - "
            f"{suffix} requires {' or '.join(allowed)}",
        )


def _parse_one(r: Reader) -> ModelConfig:
    r.reject_unknown(*_MODEL_KEYS)

    model_id = r.identifier(
        "id", ID_PATTERN,
        "letters, digits, hyphens, and underscores only (cameras reference this id)",
    )
    name = r.string("name")
    version = r.string("version")
    path_value = r.string("path")
    device = r.enum("device", DEVICES)
    classes = r.string_list("classes")
    confidence_threshold = r.number("confidence_threshold", minimum=0.0, maximum=1.0)
    iou_threshold = r.number("iou_threshold", minimum=0.0, maximum=1.0)
    before_size = r.error_count
    input_size = r.int_pair("input_size", minimum=1)
    input_size_ok = r.error_count == before_size

    use_tracker = r.boolean("use_tracker")
    tracker = r.string("tracker")

    before_half = r.error_count
    half = r.boolean("half")
    half_ok = r.error_count == before_half

    _check_format(r, path_value, device)

    # Warnings are only meaningful when the value they describe actually
    # parsed - a failed read returns a placeholder, and warning about that
    # placeholder would point the operator at the wrong problem.
    if input_size_ok and any(v % _STRIDE for v in input_size):
        r.warn(
            "input_size",
            f"{input_size} is not a multiple of {_STRIDE} - YOLO will resize internally",
        )

    if half_ok and half and path_value and device:
        if device not in ("cuda", "auto"):
            r.warn("half", f"half has no effect on device '{device}' and will be ignored")
        elif Path(path_value).suffix.lower() == ".engine":
            r.warn("half", "half has no effect on a .engine - precision is fixed at export")

    return ModelConfig(
        id=model_id,
        name=name,
        version=version,
        path=path_value,
        device=device,
        classes=classes,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        input_size=input_size,
        use_tracker=use_tracker,
        tracker=tracker,
        half=half,
    )


def parse_all(raw: Any) -> dict[str, ModelConfig]:
    """
    Parse and validate every model in models.yaml, keyed by id.

    File-existence checks for `path` and `tracker` are NOT done here. A model
    that no enabled camera uses is never loaded, so declaring more models than
    the device has weight files for is legitimate - the sample does exactly
    that. Those checks live in _validate, where cameras are in scope.
    """
    root = Reader(_FILE, {})

    if not isinstance(raw, list):
        root.error("models", f"expected a list of models, got {describe(raw)}")
        root.raise_if_errors()
    if not raw:
        root.error("models", "at least one model must be defined")
        root.raise_if_errors()

    models: dict[str, ModelConfig] = {}

    for i, item in enumerate(raw):
        path = f"models[{i}]"
        model = _parse_one(root.child(path, item))

        if model.id:
            if model.id in models:
                root.error(f"{path}.id", f"duplicate model id '{model.id}'")
            else:
                models[model.id] = model

    root.raise_if_errors()
    return models
