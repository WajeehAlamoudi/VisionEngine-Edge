from __future__ import annotations

import logging
import time

import cv2
import yaml

from .overlay import draw_completed_zones, draw_hud, draw_polygon_in_progress, draw_controls
from .stream import CameraStream

log = logging.getLogger(__name__)

CONTROLS = [
    "Q - quit  |  S - save & print YAML  |  Z - toggle existing zones",
    "N - finish zone & name it  |  U - undo last point  |  Left click - add point",
]


def run(source: str | int, existing_zones=None, title: str = "VisionEngine - Zone Builder") -> None:
    """
    mode: zones - click to draw zone polygons on the live frame.

    Controls:
      Left click  add a point to the current polygon
      U           undo last point
      N           finish current polygon - prompts for zone name in terminal
      Z           toggle overlay of existing zones from cameras.yaml
      S           print final YAML block to terminal
      Q           quit
    """
    stream = CameraStream(source)
    if not stream.open():
        return

    log.info("stream ready  %dx%d", stream.width, stream.height)
    log.info("left-click to add points | N finish zone | S save YAML | Q quit")

    # ── state ─────────────────────────────────────────────────────────────────
    current_points: list[list[int]] = []   # points being drawn right now
    completed_zones: list[dict] = []       # {name, points} - finished zones
    mouse_pos: list[int] = [0, 0]
    show_existing = bool(existing_zones)

    fps = 0.0
    t_last = time.monotonic()
    frame_count = 0

    # ── mouse callback ────────────────────────────────────────────────────────
    def on_mouse(event, x, y, flags, param):
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append([x, y])
            log.info("point %d: [%d, %d]", len(current_points), x, y)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)

    # Grab a static background frame — reused each loop so the tool stays open
    # even if the RTSP stream drops mid-session (common with H.264+ NVRs).
    static_frame = stream.read()
    if static_frame is None:
        log.error("could not grab a frame to draw on")
        stream.release()
        cv2.destroyAllWindows()
        return
    stream.release()
    log.info("frame captured - stream closed. draw zones, press S to save.")

    while True:
        frame = static_frame.copy()

        # ── draw layers ───────────────────────────────────────────────────────
        if show_existing and existing_zones:
            from .overlay import draw_zones
            draw_zones(frame, existing_zones)

        draw_completed_zones(frame, completed_zones)
        draw_polygon_in_progress(frame, current_points, tuple(mouse_pos))

        status = f"Zones: {len(completed_zones)}  |  Points: {len(current_points)}"
        draw_hud(frame, fps, stream.width, stream.height, extras=[status])
        draw_controls(frame, CONTROLS)

        cv2.imshow(title, frame)
        key = cv2.waitKey(50) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("u"):
            if current_points:
                removed = current_points.pop()
                log.info("undo - removed point [%d, %d]", removed[0], removed[1])

        elif key == ord("n"):
            if len(current_points) < 3:
                log.warning("need at least 3 points to complete a zone")
            else:
                name = input(f"  zone name (zone_{len(completed_zones) + 1}): ").strip()
                if not name:
                    name = f"zone_{len(completed_zones) + 1}"
                completed_zones.append({"name": name, "points": list(current_points)})
                log.info("zone '%s' saved with %d points", name, len(current_points))
                current_points = []

        elif key == ord("z"):
            show_existing = not show_existing
            log.info("existing zones overlay: %s", "ON" if show_existing else "OFF")

        elif key == ord("s"):
            _print_yaml(completed_zones, stream.width, stream.height)

    stream.release()
    cv2.destroyAllWindows()

    if completed_zones:
        _print_yaml(completed_zones, stream.width, stream.height)


def _print_yaml(zones: list[dict], width: int, height: int) -> None:
    if not zones:
        log.warning("no zones to save yet")
        return

    # Raw print intentional — output must be clean YAML the user can copy directly
    print(f"\n# Camera resolution: {width} x {height}")
    print("# Paste this under your camera entry in cameras.yaml:\n")
    output = {"zones": [{"name": z["name"], "polygon": z["points"]} for z in zones]}
    print(yaml.dump(output, default_flow_style=None, sort_keys=False).rstrip())
    print()
