from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _heartbeat_row(health: dict) -> dict:
    return {
        "device_id":        health["device_id"],
        "name":             health["name"],
        "location":         health["location"],
        "status":           health["status"],
        "cameras_active":   health["cameras_active"],
        "cameras_error":    health["cameras_error"],
        "detections_total": sum(c["detections_total"] for c in health["cameras"]),
        "buffer_pending":   health["buffer"]["pending"],
        "uptime_seconds":   health["uptime_seconds"],
        "ts":               health["ts"],
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
