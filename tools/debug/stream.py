from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

# Where saved frames land, relative to wherever the tool was run from.
CAPTURE_DIR = "captures"


def save_frame(frame, prefix: str = "capture", directory: str = CAPTURE_DIR) -> Path:
    """
    Write one frame to a timestamped JPEG and return where it went.

    Pass the raw frame, never the one the overlay has been drawn on. These are
    mostly taken to be uploaded as a camera's background image, and a capture
    with the HUD and boxes burned into it is useless for that.

    Saving from this tool also keeps the image the same size as the stream the
    detections come from, which is what makes the normalised anchor points line
    up with it. A still taken from a different stream of the same camera can
    have a different aspect ratio, and then every point sits in the wrong place.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
    cv2.imwrite(str(path), frame)
    return path.resolve()

_RTSP_FFMPEG_FLAGS = [
    "-rtsp_transport",  "tcp",
    "-err_detect",      "ignore_err",
    "-probesize",       "50000000",
    "-analyzeduration", "50000000",
]

_RTSP_CV2_OPTIONS = (
    "rtsp_transport;tcp"
    "|fflags;+discardcorrupt+genpts"
    "|probesize;50000000"
    "|analyzeduration;50000000"
)


class CameraStream:
    """Opens a camera source and provides frames at the native resolution."""

    def __init__(self, source: str | int) -> None:
        self._source = source
        self._cap: cv2.VideoCapture | None = None
        self.width = 0
        self.height = 0
        self.first_frame = None

    def _is_rtsp(self) -> bool:
        s = str(self._source)
        return s.startswith("rtsp://") or s.startswith("rtsps://")

    def open(self) -> bool:
        src = int(self._source) if str(self._source).isdigit() else self._source

        # ── local device ──────────────────────────────────────────────────────
        if not self._is_rtsp():
            self._cap = cv2.VideoCapture(src)
            if not self._cap.isOpened():
                log.error("cannot open device: %s", src)
                return False
            ret, frame = self._cap.read()
            if not ret or frame is None:
                log.error("no frame from device: %s", src)
                return False
            self.width       = frame.shape[1]
            self.height      = frame.shape[0]
            self.first_frame = frame.copy()
            log.info("device ready  %dx%d", self.width, self.height)
            return True

        # ── RTSP ─────────────────────────────────────────────────────────────
        # Step 1: grab one clean frame via ffmpeg -vframes 1 → JPEG.
        # This reaches AVCodecContext (-err_detect ignore_err) so H.264+/H.265+
        # non-standard SPS/VPS are tolerated. Much simpler than a raw pipe.
        log.info("grabbing frame ...")
        frame = _grab_single_frame(src)
        if frame is None:
            log.error("could not grab frame from: %s", src)
            return False

        self.first_frame = frame
        self.width       = frame.shape[1]
        self.height      = frame.shape[0]
        log.info("frame ready  %dx%d", self.width, self.height)

        # Step 2: open cv2.VideoCapture for continuous read() calls.
        # Viewer and inference_viewer need this; zone_builder releases immediately.
        #
        # The variable is process-global and OpenCV reads it when the capture is
        # constructed, so it is put back afterwards rather than left set. This
        # tool opens one camera at a time and needs no lock for it, unlike the
        # runtime in core/model/detector/ultralytics/, which opens several
        # concurrently.
        previous = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _RTSP_CV2_OPTIONS
        try:
            self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        finally:
            if previous is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            log.warning("continuous capture unavailable — first_frame still valid")

        return True

    def read(self):
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None


def _grab_single_frame(url: str):
    """Run ffmpeg -vframes 1 and return the frame as a BGR numpy array."""
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", *_RTSP_FFMPEG_FLAGS, "-i", url,
             "-vframes", "1", "-q:v", "2", tmp],
            capture_output=True,
            timeout=30,
        )
        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            return None
        return cv2.imread(tmp)
    except Exception as exc:
        log.error("ffmpeg grab failed: %s", exc)
        return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
