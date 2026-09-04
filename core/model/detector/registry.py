from __future__ import annotations

from typing import TYPE_CHECKING

from core.config import CameraConfig, ModelConfig
from core.config.model import RUNTIMES
from .base import CameraRuntime, Detector
from .ultralytics.detector import UltralyticsDetector

if TYPE_CHECKING:                       # only for the signature below; importing
    from ..runner import ModelRunner    # it for real would close a cycle

# models.yaml `runtime:` value → detector, for runtimes that go through the
# shared model layer. A runtime that opens the camera itself and runs its own
# inference — DeepStream — never reaches here and needs no entry.
_DETECTORS: dict[str, type[Detector]] = {
    "ultralytics": UltralyticsDetector,
}


def build_detector(cfg: ModelConfig, backend: str | None = None) -> Detector:
    """
    The detector for a model, from cfg.runtime unless one is named explicitly.

    Each backend imports its SDK lazily, inside its own load()/infer() — never
    at module top level — so a machine missing one runtime's dependencies can
    still run every other.
    """
    cls = _DETECTORS.get(backend or cfg.runtime)
    if cls is None:
        raise RuntimeError(
            f"runtime '{backend or cfg.runtime}' is not handled by the model "
            f"layer — available: {', '.join(_DETECTORS)}"
        )
    return cls(cfg)


def build_camera_runtime(
        cam: CameraConfig,
        model: ModelConfig,
        runner: "ModelRunner | None",
        with_frames: bool = False,
) -> CameraRuntime:
    """
    The video path for one camera, chosen by its model's runtime.

    `runner` is supplied for runtimes that go through the model layer and is
    None for those that do not — ModelRegistry builds one only when
    core.config.model.needs_model_runner() says so.

    with_frames asks for decoded pixels alongside the detections. Runtimes that
    already hold frames ignore it; one that keeps them on the GPU has to add a
    conversion and a copy per frame, so it is requested only by things that
    draw — the debug overlay — and never by the running pipeline.
    """
    if model.runtime == "deepstream":
        # Imported here, not at module level: a Raspberry Pi or a Mac must be
        # able to import this registry without the DeepStream SDK present.
        from .deepstream.runtime import DeepStreamCameraRuntime
        return DeepStreamCameraRuntime(cam, model, with_frames=with_frames)

    if model.runtime == "ultralytics":
        from .ultralytics.runtime import UltralyticsCameraRuntime
        if runner is None:
            raise RuntimeError(
                f"camera '{cam.id}': runtime '{model.runtime}' needs a model "
                f"runner and none was built"
            )
        return UltralyticsCameraRuntime(cam, runner)

    raise RuntimeError(
        f"camera '{cam.id}': unknown runtime '{model.runtime}' — "
        f"available: {', '.join(RUNTIMES)}"
    )
