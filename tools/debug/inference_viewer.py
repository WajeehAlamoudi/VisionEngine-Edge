from __future__ import annotations

import logging
import time

import cv2

from core.config import AppConfig, CameraConfig
from core.config.model import needs_model_runner
from core.model import ModelRunner
from core.model.detector import SourceUnavailable, build_camera_runtime
from .overlay import draw_detections, draw_hud, draw_zones, draw_controls

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

    # The same camera runtime main.py uses, so what is drawn here is what the
    # device actually does — including which decoder the runtime chose.
    log.info("loading model '%s' from %s ...", cam.model_id, model_cfg.path)
    runner = None
    if needs_model_runner(model_cfg):
        runner = ModelRunner(
            cfg=model_cfg,
            use_tracker=model_cfg.use_tracker,
            tracker=model_cfg.tracker,
        )
        runner.load()

    stream = build_camera_runtime(cam, model_cfg, runner)
    log.info("model ready — opening camera '%s'", camera_id)
    try:
        stream.open()
    except Exception as exc:
        log.error("cannot open camera '%s': %s", camera_id, exc)
        return

    width, height = stream.frame_size
    log.info("stream ready  %dx%d", width, height)
    log.info("classes: %s", cam.classes)
    log.info("zones:   %s", [z.name for z in cam.zones] or "none (full frame)")
    log.info("Q quit  |  Z toggle zones  |  D toggle detections")

    # Consecutive failed reads tolerated before the viewer gives up. A stream
    # that hiccups should not close the window.
    _MAX_MISSES = 30

    show_zones      = bool(cam.zones)
    show_detections = True

    fps = 0.0
    misses = 0
    inf_times: list[float] = []
    last_frame = None
    last_results: list = []

    while True:
        t0 = time.monotonic()
        inferred = True
        try:
            f, last_results = stream.read()
        except SourceUnavailable as exc:
            # One dropped frame is not the end of the stream. Keep showing the
            # last good one, exactly as CameraPipeline retries rather than
            # giving up, and only quit once it stays down.
            misses += 1
            if misses > _MAX_MISSES:
                log.warning("stream gone after %d consecutive misses: %s", misses, exc)
                break
            log.debug("frame miss %d/%d: %s", misses, _MAX_MISSES, exc)
            f, inferred = None, False
        except Exception as exc:
            log.error("inference error: %s", exc)
            f, last_results, inferred = None, [], False
        else:
            misses = 0

        if f is not None:
            last_frame = f
        elif last_frame is None:
            # A runtime that keeps frames on the GPU never returns one, so
            # there is nothing to draw on.
            log.error(
                "runtime '%s' returns no frame — the overlay needs pixels and "
                "this backend keeps them on the GPU", model_cfg.runtime)
            break

        frame = last_frame.copy()

        if inferred:
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
        draw_hud(frame, fps, width, height, extras=extras)
        draw_controls(frame, CONTROLS)

        cv2.imshow(title, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("z"):
            show_zones = not show_zones
        elif key == ord("d"):
            show_detections = not show_detections

    stream.close()
    cv2.destroyAllWindows()
