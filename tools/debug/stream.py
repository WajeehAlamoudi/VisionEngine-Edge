from __future__ import annotations

import logging
import os
import cv2

log = logging.getLogger(__name__)


class CameraStream:
    """Opens a camera source and provides frames at the native resolution."""

    def __init__(self, source: str | int) -> None:
        self._source = source
        self._cap: cv2.VideoCapture | None = None
        self.width = 0
        self.height = 0

    def open(self) -> bool:
        src = int(self._source) if str(self._source).isdigit() else self._source
        # Force TCP + tolerate non-standard SPS/PPS (H.264+, Hikvision, Dahua, etc.)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;+discardcorrupt+genpts"
        )
        self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            log.error("cannot open source: %s", self._source)
            return False

        # Decode frames until we get a valid one — grab() alone won't sync a
        # non-standard H.264+ stream that has bad SPS headers on early packets.
        log.info("syncing decoder ...")
        for _ in range(120):
            ret, frame = self._cap.read()
            if ret and frame is not None and frame.size > 0:
                self.width  = frame.shape[1]
                self.height = frame.shape[0]
                break
        else:
            log.warning("no valid frame decoded from: %s", self._source)
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
