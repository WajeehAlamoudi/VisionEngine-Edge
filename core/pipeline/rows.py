from __future__ import annotations

from datetime import datetime, timezone

from core.rules import DetectionEvent


def detection_row(event: DetectionEvent) -> dict:
    """Build a raw detection DB row from a DetectionEvent."""
    row = {
        "camera_id":     event.camera_id,
        "track_id":      event.track_id,
        "class":         event.class_name,
        "confidence":    round(event.confidence, 4),
        "bbox_x1":       int(event.bbox[0]),
        "bbox_y1":       int(event.bbox[1]),
        "bbox_x2":       int(event.bbox[2]),
        "bbox_y2":       int(event.bbox[3]),
        "anchor_x":      event.anchor_x,
        "anchor_y":      event.anchor_y,
        "anchor_x_norm": event.anchor_x_norm,
        "anchor_y_norm": event.anchor_y_norm,
        "frame_w":       event.frame_w,
        "frame_h":       event.frame_h,
        "zone":          event.zone,
        "ts":            event.capture_ts,
    }
    if event.attributes:
        row.update(event.attributes)
    return row


def notification_row(match) -> dict:
    """Build a notification DB row from a RuleMatch."""
    det = match.detection
    return {
        "rule_name":  match.rule.name,
        "class":      det.class_name,
        "camera_id":  det.camera_id,
        "zone":       det.zone,
        "confidence": round(det.confidence, 4),
        "severity":   match.rule.severity,
        "message":    match.message,
        "ts":         det.capture_ts,
    }


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
