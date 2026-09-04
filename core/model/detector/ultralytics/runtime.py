from __future__ import annotations

import logging
import os
import threading
import time

import cv2

from typing import TYPE_CHECKING

from core.config import CameraConfig
from ...types import InferenceResult

if TYPE_CHECKING:
    from ...runner import ModelRunner
from ..base import CameraRuntime, SourceUnavailable

log = logging.getLogger(__name__)

# OpenCV takes FFMPEG options from this environment variable, read when a
# capture is constructed. It is process-global, and cameras open concurrently
# in the pipeline's executor, so setting it without a lock lets one camera's
# options apply to another's open — an RTSP camera leaking rtsp_transport onto
# a file camera that started a moment later.
_FFMPEG_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_FFMPEG_ENV_LOCK = threading.Lock()

# TCP + discard-corrupt handles H.264, H.265, and H.264+ (the non-standard SPS
# headers Hikvision and Dahua emit). The large probe values give ffmpeg enough
# of the stream to identify it before giving up.
_RTSP_OPTIONS = (
    "rtsp_transport;tcp"
    "|fflags;+discardcorrupt+genpts"
    "|probesize;50000000"
    "|analyzeduration;50000000"
)


def _open_capture(source: str | int) -> cv2.VideoCapture:
    """
    Open a source with the FFMPEG options that suit it, and only those.

    The lock spans setting the variable and constructing the capture, because
    that construction is when OpenCV reads it. Whatever was there before is put
    back, so an operator who set the variable deliberately keeps it, and no
    camera inherits another's options.
    """
    if isinstance(source, int):
        # A USB index wants OpenCV's default backend — V4L2 on Linux. FFMPEG is
        # for URLs and files, and none of the options above mean anything here.
        return cv2.VideoCapture(source)

    options = _RTSP_OPTIONS if str(source).startswith(("rtsp://", "rtsps://")) else None

    with _FFMPEG_ENV_LOCK:
        previous = os.environ.get(_FFMPEG_ENV)
        if options is None:
            os.environ.pop(_FFMPEG_ENV, None)
        else:
            os.environ[_FFMPEG_ENV] = options
        try:
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        finally:
            if previous is None:
                os.environ.pop(_FFMPEG_ENV, None)
            else:
                os.environ[_FFMPEG_ENV] = previous


class UltralyticsCameraRuntime(CameraRuntime):
    """
    OpenCV capture feeding a shared ModelRunner.

    The runner is not owned here: ModelRegistry decides whether cameras share
    one, so several of these can hold the same instance. That is why the model
    layer stayed frame-in/detections-out — a runner that had swallowed a camera
    source could not be shared by two of them.
    """

    def __init__(self, cam: CameraConfig, runner: "ModelRunner") -> None:
        super().__init__(cam)
        self._runner = runner
        self._cap: cv2.VideoCapture | None = None
        self._size: tuple[int, int] = (0, 0)
        self._frame_interval = 1.0 / cam.fps_target if cam.fps_target else 0.0
        self._last_inference = 0.0

    def open(self) -> None:
        """
        Open whichever of the three source forms cameras.yaml allows: a USB
        device index, a stream URL, or a video file.
        """
        cap = _open_capture(self._cam.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source: {self._cam.source}")

        # Decode until a valid frame arrives — grab() alone will not recover a
        # non-standard H.264+ stream with bad SPS headers.
        w = h = 0
        for _ in range(120):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                break
        if w == 0:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._cap = cap
        self._size = (w, h)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._size

    def read(self) -> tuple[object | None, list[InferenceResult]]:
        """
        Read the next frame, and infer on it only when fps_target allows.

        Frames are read and discarded rather than left unread. OpenCV keeps
        decoding into its buffer whether or not this reads, so skipping the
        read would not save the decode — it would only mean the next frame
        collected is a stale one. Draining keeps what does reach the model
        current, which is the behaviour this loop has always had.
        """
        if self._cap is None:
            raise RuntimeError("read() before open()")

        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise SourceUnavailable("frame read failed")

            now = time.time()
            if now - self._last_inference >= self._frame_interval:
                self._last_inference = now
                return frame, self._runner.run(frame, self._cam.classes)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
