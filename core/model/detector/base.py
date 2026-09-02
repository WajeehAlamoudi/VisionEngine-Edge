from __future__ import annotations

from abc import ABC, abstractmethod

from core.config import ModelConfig
from ..types import InferenceResult


class Detector(ABC):
    """
    Stateless, detection-only backend. Safe to share between cameras that use
    the same model_id — carries no per-camera state (unlike Tracker).

    infer() takes a raw frame and the camera's active class list, and returns
    detections with track_id always None — tracking is a separate concern,
    added by the tracker/ layer when a camera has use_tracker=True.

    A backend that cannot separate the two sets tracks_internally (see below),
    which relaxes both statements for that backend alone.
    """

    # True when the backend assigns track_id itself, so ModelRunner must not
    # build a Tracker on top — the case when detection and tracking are fused
    # into one pipeline that cannot be taken apart, as with DeepStream's
    # nvinfer and nvtracker sharing a buffer pass.
    #
    # Whether that fusion is active can depend on config, so this is read
    # through registry.tracks_internally(cfg) rather than off the class alone.
    tracks_internally: bool = False

    # False when one instance cannot serve several cameras, so ModelRegistry
    # must give each its own. Deliberately separate from tracks_internally:
    # tracking state is one reason to be unshareable, but not the only one. A
    # backend holding a pipeline negotiated for a fixed frame size is
    # unshareable whether or not it tracks, because a second camera at another
    # resolution cannot use it.
    shareable: bool = True

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
        objects the interpreter reclaims on exit. Backends that own something
        outside the interpreter — a running GStreamer pipeline, a device
        handle — override it. Must be safe to call more than once, and safe to
        call on a detector whose load() failed.
        """
        return
