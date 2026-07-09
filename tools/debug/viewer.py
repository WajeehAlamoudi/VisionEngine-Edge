from __future__ import annotations

import logging
import time

import cv2

from .overlay import draw_hud, draw_controls
from .stream import CameraStream

log = logging.getLogger(__name__)

CONTROLS = [
    "Q - quit",
]


def run(source: str | int, title: str = "VisionEngine - View") -> None:
    """mode: view - raw stream with resolution and FPS display."""
    stream = CameraStream(source)
    if not stream.open():
        return

    log.info("stream ready  %dx%d  |  press Q to quit", stream.width, stream.height)

    fps = 0.0
    t_last = time.monotonic()
    frame_count = 0
    last_frame = None

    while True:
        f = stream.read()
        if f is not None:
            last_frame = f
            frame_count += 1
        elif last_frame is None:
            log.warning("stream ended")
            break

        frame = last_frame.copy()
        now = time.monotonic()
        elapsed = now - t_last
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            t_last = now

        draw_hud(frame, fps, stream.width, stream.height)
        draw_controls(frame, CONTROLS)

        cv2.imshow(title, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.release()
    cv2.destroyAllWindows()
