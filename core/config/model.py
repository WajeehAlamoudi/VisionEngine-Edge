from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .strict import ID_PATTERN, Reader, describe

_FILE = "models.yaml"

_MODEL_KEYS = (
    "id", "name", "version", "path", "runtime", "accelerator", "classes",
    "confidence_threshold", "iou_threshold", "input_size",
    "use_tracker", "tracker", "half", "ds_infer_config",
)

# Which code path runs the model. Must stay in sync with the packages under
# core/model/detector/, one per runtime. Not every name appears in _DETECTORS
# there: a runtime that opens the camera itself has no Detector at all, only a
# CameraRuntime. Duplicated rather than imported because those modules reach
# their SDKs, and config validation must stay free of heavy deps.
RUNTIMES = ("ultralytics", "deepstream")

# Runtimes that open the camera themselves, running detection and tracking in
# one pipeline of their own (see core/model/detector/). A model on one of these
# never reaches the shared model layer, so ModelRegistry builds it no runner.
SELF_CAPTURING_RUNTIMES = ("deepstream",)

# Which hardware the runtime executes on.
ACCELERATORS = ("auto", "cpu", "cuda", "mps", "coreml")

# Not every accelerator makes sense for every runtime. DeepStream is an NVIDIA
# SDK and has nowhere else to run.
_RUNTIME_ACCELERATORS: dict[str, tuple[str, ...]] = {
    "ultralytics": ("auto", "cpu", "cuda", "mps", "coreml"),
    "deepstream":  ("cuda",),
}

# Which accelerators each (runtime, weight format) pair can actually run on.
# Keyed by both because the same extension means different things to different
# runtimes: a .engine is a YOLO export to Ultralytics and a network for nvinfer
# to DeepStream, and loading one as the other fails deep inside the SDK rather
# than at startup.
#
# .engine deliberately excludes "auto": a TensorRT engine cannot load on CPU,
# and auto resolves to CPU whenever CUDA is unavailable, turning a missing GPU
# into an obscure runtime failure instead of a clear config error.
_FORMAT_SUPPORT: dict[tuple[str, str], tuple[str, ...]] = {
    ("ultralytics", ".pt"):        ("auto", "cpu", "cuda", "mps"),
    ("ultralytics", ".onnx"):      ("auto", "cpu", "cuda", "mps"),
    ("ultralytics", ".engine"):    ("cuda",),
    ("ultralytics", ".mlpackage"): ("coreml",),
    ("ultralytics", ".tflite"):    ("cpu",),
    ("deepstream",  ".engine"):    ("cuda",),
    ("deepstream",  ".onnx"):      ("cuda",),
}

# device was one key doing two jobs: naming the hardware (cpu/cuda/mps) and the
# runtime (coreml/deepstream). Kept only to explain the split, since
# reject_unknown would otherwise report it as an unrecognised key and say
# nothing about what replaced it.
_DEVICE_MIGRATION = {
    "auto":       ("ultralytics", "auto"),
    "cpu":        ("ultralytics", "cpu"),
    "cuda":       ("ultralytics", "cuda"),
    "mps":        ("ultralytics", "mps"),
    "coreml":     ("ultralytics", "coreml"),
    "deepstream": ("deepstream", "cuda"),
}

# YOLO models use a stride of 32, so other sizes get resized internally.
_STRIDE = 32


@dataclass
class ModelConfig:
    id: str
    name: str
    version: str
    path: str
    runtime: str            # ultralytics | deepstream - which code path runs it
    accelerator: str        # auto | cpu | cuda | mps | coreml - what it runs on
    classes: list[str]      # what this deployment expects the model to detect
    confidence_threshold: float
    iou_threshold: float
    input_size: list[int]   # [width, height]
    use_tracker: bool        # true = tracking ON → track_id populated per object
    tracker: str             # path to the tracker params file — a BoT-SORT yaml on
                              # the ultralytics runtime, and on deepstream the
                              # nvtracker config, which also chooses the algorithm
    half: bool               # FP16 inference — only applied when the accelerator
                              # resolves to cuda; ignored on cpu/mps, where it gives
                              # no benefit (see accelerator.py)
    ds_infer_config: str | None = None
                             # nvinfer config file. Required for runtime: deepstream,
                             # meaningless for every other runtime.


def needs_model_runner(model: "ModelConfig") -> bool:
    """
    Whether this model is executed through the shared model layer.

    False for a runtime that owns its own capture and inference. Answered from
    config alone so neither the model layer nor the runtime layer has to import
    the other to find out.
    """
    return model.runtime not in SELF_CAPTURING_RUNTIMES


def _check_format(r: Reader, path_value: str, runtime: str, accelerator: str) -> None:
    """The weight format must be one this runtime can load on this accelerator."""
    if not path_value or not runtime or not accelerator:
        return

    suffix = Path(path_value).suffix.lower()
    known = {fmt for _, fmt in _FORMAT_SUPPORT}
    if suffix not in known:
        r.error(
            r.path_of("path"),
            f"unrecognised model format '{suffix}' - "
            f"expected one of {', '.join(sorted(known))}",
        )
        return

    allowed = _FORMAT_SUPPORT.get((runtime, suffix))
    if allowed is None:
        runners = sorted(rt for rt, fmt in _FORMAT_SUPPORT if fmt == suffix)
        r.error(
            r.path_of("path"),
            f"runtime '{runtime}' cannot load a {suffix} model - "
            f"{suffix} is loaded by {' or '.join(runners)}",
        )
        return

    if accelerator not in allowed:
        r.error(
            r.path_of("accelerator"),
            f"accelerator '{accelerator}' cannot run a {suffix} model on the "
            f"{runtime} runtime - it requires {' or '.join(allowed)}",
        )


def _check_migration(r: Reader) -> None:
    """
    Explain the device -> runtime + accelerator split.

    reject_unknown would report 'device' as an unrecognised key and say nothing
    about what replaced it, which is a poor way to meet a rename.
    """
    old = r.raw_value("device")
    if old is None:
        return
    mapped = _DEVICE_MIGRATION.get(old if isinstance(old, str) else "")
    if mapped:
        runtime, accelerator = mapped
        hint = (f"replace 'device: {old}' with "
                f"'runtime: {runtime}' and 'accelerator: {accelerator}'")
    else:
        hint = ("replace it with a runtime (" + " | ".join(RUNTIMES) +
                ") and an accelerator (" + " | ".join(ACCELERATORS) + ")")
    r.error(
        r.path_of("device"),
        f"'device' was one key doing two jobs and has been split - {hint}",
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
    _check_migration(r)
    runtime = r.enum("runtime", RUNTIMES)
    accelerator = r.enum("accelerator", ACCELERATORS)
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

    # Optional rather than required-with-null: it configures nvinfer, so it is
    # meaningless on every runtime except deepstream, and demanding an explicit
    # null in each Pi/Mac/CPU model entry would be noise.
    ds_infer_config = r.optional_string("ds_infer_config")

    _check_format(r, path_value, runtime, accelerator)

    allowed_accelerators = _RUNTIME_ACCELERATORS.get(runtime)
    if allowed_accelerators and accelerator not in allowed_accelerators:
        r.error(
            r.path_of("accelerator"),
            f"runtime '{runtime}' cannot run on accelerator '{accelerator}' - "
            f"it supports {' or '.join(allowed_accelerators)}",
        )

    if runtime == "deepstream" and not ds_infer_config:
        r.error(
            r.path_of("ds_infer_config"),
            "required for runtime 'deepstream' - nvinfer is configured by this "
            "file (engine, network shape, class count, clustering), and there "
            "is no default. Copy config/config_sample/deepstream_infer.sample.txt",
        )
    elif runtime != "deepstream" and ds_infer_config:
        r.warn(
            "ds_infer_config",
            f"ignored on runtime '{runtime}' - it only configures nvinfer, "
            f"which only the deepstream runtime uses",
        )


    # Warnings are only meaningful when the value they describe actually
    # parsed - a failed read returns a placeholder, and warning about that
    # placeholder would point the operator at the wrong problem.
    if input_size_ok and any(v % _STRIDE for v in input_size):
        r.warn(
            "input_size",
            f"{input_size} is not a multiple of {_STRIDE} - YOLO will resize internally",
        )

    if half_ok and half and path_value and accelerator:
        if accelerator not in ("cuda", "auto"):
            r.warn("half",
                   f"half has no effect on accelerator '{accelerator}' and will be ignored")
        elif Path(path_value).suffix.lower() == ".engine":
            r.warn("half", "half has no effect on a .engine - precision is fixed at export")

    return ModelConfig(
        id=model_id,
        name=name,
        version=version,
        path=path_value,
        runtime=runtime,
        accelerator=accelerator,
        classes=classes,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        input_size=input_size,
        use_tracker=use_tracker,
        tracker=tracker,
        half=half,
        ds_infer_config=ds_infer_config,
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
