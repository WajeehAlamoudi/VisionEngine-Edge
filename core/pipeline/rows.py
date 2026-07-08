from __future__ import annotations

from datetime import datetime, timezone

from core.rules import DetectionEvent


def detection_row(event: DetectionEvent) -> dict:
    """Build a raw detection DB row from a DetectionEvent."""
    x1, y1, x2, y2 = (int(v) for v in event.bbox)
    row = {
        # ── identity ──────────────────────────────────────────────────────────
        "camera_id":     event.camera_id,
        "camera_name":   event.camera_name,
        "model_id":      event.model_id,
        "track_id":      event.track_id,

        # ── detection ─────────────────────────────────────────────────────────
        "class":         event.class_name,
        "confidence":    round(event.confidence, 4),

        # ── bounding box — absolute pixels ────────────────────────────────────
        "bbox_x1":       x1,
        "bbox_y1":       y1,
        "bbox_x2":       x2,
        "bbox_y2":       y2,
        "bbox_w":        x2 - x1,
        "bbox_h":        y2 - y1,

        # ── anchor point — pixel + normalized ─────────────────────────────────
        "anchor_x":      event.anchor_x,
        "anchor_y":      event.anchor_y,
        "anchor_x_norm": event.anchor_x_norm,
        "anchor_y_norm": event.anchor_y_norm,

        # ── frame context ─────────────────────────────────────────────────────
        "frame_w":       event.frame_w,
        "frame_h":       event.frame_h,
        "zone":          event.zone,

        # ── temporal ──────────────────────────────────────────────────────────
        "ts":            event.capture_ts,
    }
    if event.attributes:
        row.update(event.attributes)
    return row


def notification_row(match) -> dict:
    """Build a notification DB row from a RuleMatch."""
    det = match.detection
    x1, y1, x2, y2 = (int(v) for v in det.bbox)
    row = {
        # ── rule ──────────────────────────────────────────────────────────────
        "rule_name":     match.rule.name,
        "severity":      match.rule.severity,
        "message":       match.message,

        # ── identity ──────────────────────────────────────────────────────────
        "camera_id":     det.camera_id,
        "camera_name":   det.camera_name,
        "model_id":      det.model_id,
        "track_id":      det.track_id,

        # ── detection ─────────────────────────────────────────────────────────
        "class":         det.class_name,
        "confidence":    round(det.confidence, 4),

        # ── bounding box — absolute pixels ────────────────────────────────────
        "bbox_x1":       x1,
        "bbox_y1":       y1,
        "bbox_x2":       x2,
        "bbox_y2":       y2,
        "bbox_w":        x2 - x1,
        "bbox_h":        y2 - y1,

        # ── anchor point — pixel + normalized ─────────────────────────────────
        "anchor_x":      det.anchor_x,
        "anchor_y":      det.anchor_y,
        "anchor_x_norm": det.anchor_x_norm,
        "anchor_y_norm": det.anchor_y_norm,

        # ── frame context ─────────────────────────────────────────────────────
        "frame_w":       det.frame_w,
        "frame_h":       det.frame_h,
        "zone":          det.zone,

        # ── temporal ──────────────────────────────────────────────────────────
        "ts":            det.capture_ts,
    }
    if det.attributes:
        row.update(det.attributes)
    return row


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
