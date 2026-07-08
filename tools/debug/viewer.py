from __future__ import annotations

import time

import cv2

from .overlay import draw_hud, draw_controls
from .stream import CameraStream

CONTROLS = [
    "Q — quit",
]


def run(source: str | int, title: str = "VisionEngine — View") -> None:
    """mode: view — raw stream with resolution and FPS display."""
    stream = CameraStream(source)
    if not stream.open():
        return

    print(f"Stream opened  {stream.width}x{stream.height}")
    print("Press Q to quit.")

    fps = 0.0
    t_last = time.monotonic()
    frame_count = 0

    while True:
        frame = stream.read()
        if frame is None:
            print("Stream ended.")
            break

        frame_count += 1
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
