from __future__ import annotations

import cv2
import numpy as np

# ── color palette (BGR) ───────────────────────────────────────────────────────
CLR_ZONE        = (255, 200,  50)   # amber  — zone polygons
CLR_ZONE_LABEL  = (255, 200,  50)
CLR_POLY_ACTIVE = ( 50, 220, 255)   # yellow — polygon in progress
CLR_POLY_POINT  = (  0, 255, 255)
CLR_POLY_EDGE   = ( 50, 220, 255)
CLR_DET_BOX     = ( 50, 220,  50)   # green  — detection bbox
CLR_DET_LABEL   = ( 50, 220,  50)
CLR_HUD         = (200, 200, 200)   # light grey — HUD text

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _text(frame, text: str, pos: tuple[int, int], color=CLR_HUD,
          scale: float = 0.55, thickness: int = 1, font=FONT) -> None:
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame, fps: float, width: int, height: int, extras: list[str] | None = None) -> None:
    """Top-left HUD: resolution, fps, and optional extra lines."""
    lines = [
        f"Resolution: {width} x {height}",
        f"FPS: {fps:.1f}",
    ]
    if extras:
        lines.extend(extras)
    for i, line in enumerate(lines):
        _text(frame, line, (10, 22 + i * 22), CLR_HUD, scale=0.55)


def draw_zones(frame, zones, alpha: float = 0.15) -> None:
    """Draw zone polygons with fill + border + name label."""
    overlay = frame.copy()
    for zone in zones:
        pts = np.array(zone.polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], CLR_ZONE)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for zone in zones:
        pts = np.array(zone.polygon, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=CLR_ZONE, thickness=2, lineType=cv2.LINE_AA)
        cx = int(np.mean([p[0] for p in zone.polygon]))
        cy = int(np.mean([p[1] for p in zone.polygon]))
        _text(frame, zone.name, (cx - 40, cy), CLR_ZONE_LABEL, scale=0.6, thickness=1, font=FONT_BOLD)


def draw_detections(frame, detections) -> None:
    """Draw bounding boxes with class name and confidence."""
    for det in detections:
        x1, y1, x2, y2 = int(det.bbox[0]), int(det.bbox[1]), int(det.bbox[2]), int(det.bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_DET_BOX, 2, lineType=cv2.LINE_AA)

        label = f"{det.class_name} {det.confidence:.2f}"
        if det.track_id is not None:
            label = f"[{det.track_id}] {label}"

        lw, lh = cv2.getTextSize(label, FONT, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), CLR_DET_BOX, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_polygon_in_progress(frame, points: list[list[int]], mouse_pos: tuple[int, int] | None = None) -> None:
    """Draw the polygon currently being built by the zone builder."""
    if not points:
        return

    pts = [tuple(p) for p in points]

    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], CLR_POLY_EDGE, 2, lineType=cv2.LINE_AA)

    if mouse_pos and len(pts) >= 1:
        cv2.line(frame, pts[-1], mouse_pos, CLR_POLY_ACTIVE, 1, lineType=cv2.LINE_AA)

    if len(pts) >= 3 and mouse_pos:
        cv2.line(frame, pts[0], mouse_pos, CLR_POLY_ACTIVE, 1, lineType=cv2.LINE_AA)

    for i, pt in enumerate(pts):
        cv2.circle(frame, pt, 5, CLR_POLY_POINT, -1, lineType=cv2.LINE_AA)
        _text(frame, str(i + 1), (pt[0] + 7, pt[1] - 7), CLR_POLY_POINT, scale=0.45)


def draw_completed_zones(frame, zones: list[dict]) -> None:
    """Draw already-completed zones from the zone builder (list of dicts with name + points)."""
    overlay = frame.copy()
    for zone in zones:
        pts = np.array(zone["points"], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], CLR_ZONE)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    for zone in zones:
        pts = np.array(zone["points"], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=CLR_ZONE, thickness=2, lineType=cv2.LINE_AA)
        cx = int(np.mean([p[0] for p in zone["points"]]))
        cy = int(np.mean([p[1] for p in zone["points"]]))
        _text(frame, zone["name"], (cx - 40, cy), CLR_ZONE_LABEL, scale=0.6, thickness=1, font=FONT_BOLD)


def draw_controls(frame, lines: list[str]) -> None:
    """Bottom-left control hint overlay."""
    h = frame.shape[0]
    for i, line in enumerate(reversed(lines)):
        _text(frame, line, (10, h - 12 - i * 20), CLR_HUD, scale=0.45)
