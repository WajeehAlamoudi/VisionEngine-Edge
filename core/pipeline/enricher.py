from __future__ import annotations

from core.config import CameraConfig
from core.model import InferenceResult
from core.rules import DetectionEvent
from core.zone import assign_zone


def enrich(
        inf: InferenceResult,
        cam: CameraConfig,
        frame_w: int,
        frame_h: int,
        capture_ts: str,
) -> DetectionEvent:
    """
    Convert a raw InferenceResult into a fully enriched DetectionEvent.

    Adds: zone assignment, anchor point, normalized coordinates,
          camera context, frame dimensions, and capture timestamp.
    """
    x1, y1, x2, y2 = inf.bbox

    # Anchor point — person = feet (bottom-center), everything else = bbox center.
    # A person close to the camera has their feet out of shot, so the box stops at
    # the frame bottom and y2 becomes the edge itself — a point no zone contains,
    # which tags the nearest people "unzoned" and hides them from any zoned rule.
    # For a clipped box the center is the closest honest position.
    cx = (x1 + x2) / 2
    feet_cut_off = frame_h and y2 >= frame_h - 1
    cy = (y1 + y2) / 2 if feet_cut_off or inf.class_name != "person" else y2

    zone = assign_zone(cx, cy, cam.zones)

    return DetectionEvent(
        track_id=inf.track_id,
        class_name=inf.class_name,
        confidence=inf.confidence,
        bbox=inf.bbox,
        model_id=cam.model_id,
        anchor_x=round(cx, 1),
        anchor_y=round(cy, 1),
        anchor_x_norm=round(cx / frame_w, 4) if frame_w else 0.0,
        anchor_y_norm=round(cy / frame_h, 4) if frame_h else 0.0,
        camera_id=cam.id,
        camera_name=cam.name,
        frame_w=frame_w,
        frame_h=frame_h,
        zone=zone,
        capture_ts=capture_ts,
    )
