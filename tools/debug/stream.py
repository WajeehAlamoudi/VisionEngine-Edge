from __future__ import annotations

import logging
import os
import subprocess
import tempfile

import cv2

log = logging.getLogger(__name__)

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
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _RTSP_CV2_OPTIONS
        self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
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
