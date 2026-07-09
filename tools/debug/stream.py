from __future__ import annotations

import json
import logging
import subprocess

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Passed to ffprobe and ffmpeg before -i.
# -err_detect ignore_err reaches AVCodecContext — the only way to tolerate
# Hikvision H.264+/H.265+ non-standard SPS/VPS that OpenCV cannot reach.
_RTSP_INPUT_FLAGS = [
    "-rtsp_transport",  "tcp",
    "-fflags",          "+discardcorrupt+genpts",
    "-err_detect",      "ignore_err",
    "-probesize",       "50000000",
    "-analyzeduration", "50000000",
]


def _probe_wh(url: str) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet",
         "-rtsp_transport", "tcp",
         "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "json",
         url],
        capture_output=True, text=True, timeout=20,
    )
    data = json.loads(result.stdout)
    s = data["streams"][0]
    return s["width"], s["height"]


class CameraStream:
    """Opens a camera source and provides frames at the native resolution."""

    def __init__(self, source: str | int) -> None:
        self._source = source
        self._cap:  cv2.VideoCapture | None = None   # local device path
        self._proc: subprocess.Popen | None = None   # RTSP via ffmpeg pipe
        self._frame_size = 0
        self.width = 0
        self.height = 0
        self.first_frame = None

    def _is_rtsp(self) -> bool:
        s = str(self._source)
        return s.startswith("rtsp://") or s.startswith("rtsps://")

    def open(self) -> bool:
        src = int(self._source) if str(self._source).isdigit() else self._source

        # ── local webcam / device ─────────────────────────────────────────────
        if not self._is_rtsp():
            self._cap = cv2.VideoCapture(src)
            if not self._cap.isOpened():
                log.error("cannot open device: %s", src)
                return False
            ret, frame = self._cap.read()
            if not ret or frame is None:
                log.warning("no valid frame from device: %s", src)
                return False
            self.width       = frame.shape[1]
            self.height      = frame.shape[0]
            self.first_frame = frame.copy()
            log.info("device ready  %dx%d", self.width, self.height)
            return True

        # ── RTSP via ffmpeg subprocess ────────────────────────────────────────
        log.info("probing stream dimensions ...")
        try:
            w, h = _probe_wh(src)
        except Exception as exc:
            log.error("ffprobe failed: %s", exc)
            return False

        self.width       = w
        self.height      = h
        self._frame_size = w * h * 3

        cmd = [
            "ffmpeg",
            *_RTSP_INPUT_FLAGS,
            "-i", src,
            "-vf", f"scale={w}:{h}",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]

        log.info("syncing decoder ...")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        # The decoder may output black frames while recovering from H.264+
        # SPS errors — drain until we get a frame with actual pixel content.
        for _ in range(60):
            raw = self._proc.stdout.read(self._frame_size)
            if len(raw) < self._frame_size:
                log.warning("no valid frame decoded from: %s", src)
                self.release()
                return False
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
            if frame.max() > 10:
                self.first_frame = frame
                log.info("decoder synced  %dx%d", w, h)
                return True

        log.warning("only black frames received from: %s", src)
        self.release()
        return False

    def read(self):
        if self._cap is not None:
            ret, frame = self._cap.read()
            return frame if ret else None
        if self._proc is not None:
            raw = self._proc.stdout.read(self._frame_size)
            if len(raw) < self._frame_size:
                return None
            return (
                np.frombuffer(raw, dtype=np.uint8)
                .reshape((self.height, self.width, 3))
                .copy()
            )
        return None

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None
