from __future__ import annotations

from core.rules import DetectionEvent


def dwell_row(camera_id: str, track_id: int, class_name: str, zone: str, entry_ts: str, exit_ts: str) -> dict:
    return {
        "camera_id": camera_id,
        "track_id":  track_id,
        "class":     class_name,
        "zone":      zone,
        "entry_ts":  entry_ts,
        "exit_ts":   exit_ts,
    }


def occupancy_row(camera_id: str, zone: str, class_name: str, count: int, ts: str) -> dict:
    return {
        "camera_id": camera_id,
        "zone":      zone,
        "class":     class_name,
        "count":     count,
        "ts":        ts,
    }


def trajectory_row(event: DetectionEvent) -> dict:
    return {
        "camera_id":     event.camera_id,
        "track_id":      event.track_id,
        "class":         event.class_name,
        "zone":          event.zone,
        "anchor_x":      event.anchor_x,
        "anchor_y":      event.anchor_y,
        "anchor_x_norm": event.anchor_x_norm,
        "anchor_y_norm": event.anchor_y_norm,
        "ts":            event.capture_ts,
    }
