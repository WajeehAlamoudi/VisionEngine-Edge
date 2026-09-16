from __future__ import annotations

import logging
import time

import cv2

from .overlay import draw_hud, draw_controls
from .stream import CameraStream, save_frame

log = logging.getLogger(__name__)

CONTROLS = [
    "Q - quit  |  S - save frame",
]


def run(source: str | int, name: str = "capture", title: str = "VisionEngine - View") -> None:
    """
    mode: view - raw stream with resolution and FPS display.

    `name` prefixes any frame saved with S, so captures from several cameras
    stay apart.
    """
    stream = CameraStream(source)
    if not stream.open():
        return

    log.info("stream ready  %dx%d  |  Q to quit, S to save a frame",
             stream.width, stream.height)

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
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            # last_frame, not frame: the latter carries the HUD.
            log.info("saved %s  (%dx%d)",
                     save_frame(last_frame, name), stream.width, stream.height)

    stream.release()
    cv2.destroyAllWindows()
