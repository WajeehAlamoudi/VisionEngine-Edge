from __future__ import annotations

from abc import ABC, abstractmethod

from core.config import CameraConfig, ModelConfig
from ..types import InferenceResult


class SourceUnavailable(Exception):
    """
    The camera gave no frame — dropped connection, decode failure, end of file.

    Distinct from an inference error on purpose: CameraPipeline waits and
    retries on this and only logs on anything else. Every runtime raises it for
    the same condition, so the pipeline needs no per-runtime knowledge.
    """


class Detector(ABC):
    """
    Stateless, detection-only backend. Safe to share between cameras that use
    the same model_id — carries no per-camera state (unlike Tracker).

    Deliberately knows nothing about cameras: it is handed a frame and returns
    boxes. Where that frame came from is the CameraRuntime's business. Keeping
    the two apart is what lets one loaded model serve several cameras.

    infer() takes a raw frame and the camera's active class list, and returns
    detections with track_id always None — tracking is a separate concern,
    added by the tracker/ layer when a camera has use_tracker=True.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def infer(self, frame, active_classes: list[str]) -> list[InferenceResult]: ...

    def close(self) -> None:
        """
        Release whatever load() acquired. Called once on shutdown.

        Not abstract, and a no-op by default: most backends hold only Python
        objects the interpreter reclaims on exit. A backend owning something
        outside the interpreter overrides it. Must be safe to call twice, and
        on a detector whose load() failed.
        """
        return


class CameraRuntime(ABC):
    """
    Turns one camera's source into detections.

    This is the video layer: connecting to the stream, decoding it, pacing it
    to fps_target, and getting boxes out. What that costs and how it is best
    done differs completely between runtimes — OpenCV decoding to host memory
    for Ultralytics, NVDEC keeping frames on the GPU for DeepStream — so each
    supplies its own instead of being handed frames by a caller that has
    already decided.

    It owns no business logic. Zones, rules, storage, alerts, health and
    collection stay in CameraPipeline, which is identical whichever runtime
    feeds it.

    Every method is plain blocking code. CameraPipeline owns the event loop and
    wraps these in an executor, so nothing here imports asyncio.
    """

    def __init__(self, cam: CameraConfig) -> None:
        self._cam = cam

    @abstractmethod
    def open(self) -> None:
        """
        Connect to the source and get ready to read.

        Raises on failure — the camera is then reported as failed and its
        pipeline stops, rather than looping on a source that will never open.
        frame_size is valid once this returns.
        """

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of the source. Known only after open()."""

    @abstractmethod
    def read(self) -> tuple[object | None, list[InferenceResult]]:
        """
        Advance to the next frame to process and return its detections.

        Pacing to fps_target happens here, because the cheapest place to skip a
        frame depends on the runtime: one drops it after decoding, another
        before.

        The frame is returned only when the runtime holds it in host memory. A
        runtime keeping frames on the GPU returns None, and anything needing
        pixels has to ask for them explicitly.

        Raises SourceUnavailable when the source produced nothing.
        """

    def close(self) -> None:
        """Release the source. Safe to call twice, and after a failed open()."""
        return
