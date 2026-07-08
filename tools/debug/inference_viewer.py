from __future__ import annotations

import time

import cv2

from core.config import AppConfig, CameraConfig
from core.model import ModelRunner
from .overlay import draw_detections, draw_hud, draw_zones, draw_controls
from .stream import CameraStream

CONTROLS = [
    "Q — quit  |  Z — toggle zones  |  D — toggle detections",
]


def run(cfg: AppConfig, camera_id: str, title: str = "VisionEngine — Inference") -> None:
    """
    mode: inference — live model inference with detection overlay and zone visualization.

    Shows:
      - Live bounding boxes with class name, confidence, track_id
      - Configured zones drawn on frame
      - Real inference FPS the device achieves
      - Resolution HUD
    """
    cam: CameraConfig | None = cfg.get_camera(camera_id)
    if cam is None:
        print(f"ERROR  camera '{camera_id}' not found in config")
        return

    model_cfg = cfg.get_model(cam.model_id)
    if model_cfg is None:
        print(f"ERROR  model '{cam.model_id}' not found in models.yaml")
        return

    print(f"Loading model '{cam.model_id}' from {model_cfg.path} ...")
    runner = ModelRunner(
        cfg=model_cfg,
        use_tracker=cam.analytics.needs_tracker,
        tracker=cam.analytics.tracker,
    )
    runner.load()
    print(f"Model ready. Opening camera '{camera_id}' source: {cam.source}")

    stream = CameraStream(cam.source)
    if not stream.open():
        return

    print(f"Stream opened  {stream.width}x{stream.height}")
    print(f"Classes: {cam.classes}")
    print(f"Zones:   {[z.name for z in cam.zones] or 'none (full frame)'}")
    print("Press Q to quit. Press Z to toggle zones. Press D to toggle detections.\n")

    show_zones      = bool(cam.zones)
    show_detections = True

    fps = 0.0
    inf_times: list[float] = []

    while True:
        frame = stream.read()
        if frame is None:
            print("Stream ended.")
            break

        t0 = time.monotonic()
        try:
            results = runner.run(frame, cam.classes)
        except Exception as exc:
            print(f"Inference error: {exc}")
            results = []
        inf_time = time.monotonic() - t0
        inf_times.append(inf_time)
        if len(inf_times) > 10:
            inf_times.pop(0)
        fps = 1.0 / (sum(inf_times) / len(inf_times)) if inf_times else 0.0

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
