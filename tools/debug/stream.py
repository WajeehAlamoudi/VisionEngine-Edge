from __future__ import annotations

import os
import cv2


class CameraStream:
    """Opens a camera source and provides frames at the native resolution."""

    def __init__(self, source: str | int) -> None:
        self._source = source
        self._cap: cv2.VideoCapture | None = None
        self.width = 0
        self.height = 0

    def open(self) -> bool:
        src = int(self._source) if str(self._source).isdigit() else self._source
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            print(f"ERROR  cannot open source: {self._source}")
            return False
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
