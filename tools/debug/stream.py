from __future__ import annotations

import logging
import os

import cv2
import ffmpegcv

log = logging.getLogger(__name__)

_RTSP_INPUT_OPTS = {
    "-rtsp_transport":    "tcp",
    "-fflags":            "+discardcorrupt+genpts",
    "-err_detect":        "ignore_err",
    "-probesize":         "50000000",
    "-analyzeduration":   "50000000",
}


class CameraStream:
    """Opens a camera source and provides frames at the native resolution."""

    def __init__(self, source: str | int) -> None:
        self._source = source
        self._cap = None
        self.width = 0
        self.height = 0
        self.first_frame = None

    def open(self) -> bool:
        src = int(self._source) if str(self._source).isdigit() else self._source

        try:
            if isinstance(src, int):
                # Local webcam / device — plain OpenCV, no RTSP issues
                self._cap = cv2.VideoCapture(src)
            else:
                # RTSP / file — ffmpegcv passes -err_detect ignore_err at codec level,
                # which fixes Hikvision H.264+/H.265+ non-standard SPS/VPS headers that
                # OpenCV's FFmpeg backend cannot reach via OPENCV_FFMPEG_CAPTURE_OPTIONS.
                self._cap = ffmpegcv.VideoCapture(
                    src,
                    pix_fmt="bgr24",
                    inputdict=_RTSP_INPUT_OPTS,
                )
        except Exception as exc:
            log.error("failed to open source %s: %s", self._source, exc)
            return False

        log.info("syncing decoder ...")
        for _ in range(120):
            ret, frame = self._cap.read()
            if ret and frame is not None and frame.size > 0:
                self.width       = frame.shape[1]
                self.height      = frame.shape[0]
                self.first_frame = frame.copy()
                break
        else:
            log.warning("no valid frame decoded from: %s", self._source)
            self.release()
            return False

        log.info("decoder synced  %dx%d", self.width, self.height)
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
