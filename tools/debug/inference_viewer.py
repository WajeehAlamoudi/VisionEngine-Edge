from __future__ import annotations

import logging
import time

import cv2

from core.config import AppConfig, CameraConfig
from core.model import ModelRunner
from .overlay import draw_detections, draw_hud, draw_zones, draw_controls
from .stream import CameraStream

log = logging.getLogger(__name__)

CONTROLS = [
    "Q - quit  |  Z - toggle zones  |  D - toggle detections",
]


def run(cfg: AppConfig, camera_id: str, title: str = "VisionEngine - Inference") -> None:
    """
    mode: inference - live model inference with detection overlay and zone visualization.

    Shows:
      - Live bounding boxes with class name, confidence, track_id
      - Configured zones drawn on frame
      - Real inference FPS the device achieves
      - Resolution HUD
    """
    cam: CameraConfig | None = cfg.get_camera(camera_id)
    if cam is None:
        log.error("camera '%s' not found in config", camera_id)
        return

    model_cfg = cfg.get_model(cam.model_id)
    if model_cfg is None:
        log.error("model '%s' not found in models.yaml", cam.model_id)
        return

    log.info("loading model '%s' from %s ...", cam.model_id, model_cfg.path)
    runner = ModelRunner(cfg=model_cfg)
    runner.load()
    log.info("model ready — opening camera '%s'", camera_id)

    stream = CameraStream(cam.source)
    if not stream.open():
        return

    log.info("stream ready  %dx%d", stream.width, stream.height)
    log.info("classes: %s", cam.classes)
    log.info("zones:   %s", [z.name for z in cam.zones] or "none (full frame)")
    log.info("Q quit  |  Z toggle zones  |  D toggle detections")

    show_zones      = bool(cam.zones)
    show_detections = True

    fps = 0.0
    inf_times: list[float] = []
    last_frame = None
    last_results: list = []

    while True:
        f = stream.read()
        stale = f is None
        if not stale:
            last_frame = f
        elif last_frame is None:
            log.warning("stream ended")
            break

        frame = last_frame.copy()

        if not stale:
            t0 = time.monotonic()
            try:
                last_results = runner.run(frame, cam.classes)
            except Exception as exc:
                log.error("inference error: %s", exc)
                last_results = []
            inf_time = time.monotonic() - t0
            inf_times.append(inf_time)
            if len(inf_times) > 10:
                inf_times.pop(0)
            fps = 1.0 / (sum(inf_times) / len(inf_times)) if inf_times else 0.0

        results = last_results

        # ── draw layers ───────────────────────────────────────────────────────
        if show_zones and cam.zones:
            draw_zones(frame, cam.zones)

        if show_detections:
            draw_detections(frame, results)

        extras = [
            f"Detections: {len(results)}",
            f"Classes: {', '.join(cam.classes)}",
            f"Zones: {'ON' if show_zones else 'OFF'}  |  Detections: {'ON' if show_detections else 'OFF'}",
        ]
        draw_hud(frame, fps, stream.width, stream.height, extras=extras)
        draw_controls(frame, CONTROLS)

        cv2.imshow(title, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("z"):
            show_zones = not show_zones
        elif key == ord("d"):
            show_detections = not show_detections

    stream.release()
    cv2.destroyAllWindows()
