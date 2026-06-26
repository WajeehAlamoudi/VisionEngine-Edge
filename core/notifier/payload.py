from __future__ import annotations


def _build_payload(match, device_id: str) -> dict:
    det = match.detection
    rule = match.rule
    return {
        "rule":        rule.name,
        "class":       det.class_name,
        "camera_id":   det.camera_id,
        "camera_name": det.camera_name,
        "zone":        det.zone,
        "confidence":  round(det.confidence, 4),
        "severity":    rule.severity,
        "message":     match.message,
        "device_id":   device_id,
        "ts":          det.capture_ts,
    }
