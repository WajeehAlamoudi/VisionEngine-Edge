from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HeartbeatConfig:
    enabled: bool
    interval_seconds: int
    table: str


@dataclass
class HealthFileConfig:
    enabled: bool
    path: str
    interval_seconds: int


@dataclass
class BufferConfig:
    path: str
    max_size_mb: int
    retry_interval_seconds: int
    delete_after_hours: int


@dataclass
class DeviceConfig:
    id: str
    name: str
    location: str
    environment: str        # production | development
    fps_target: int
    max_cameras: int
    log_level: str          # DEBUG | INFO | WARNING | ERROR
    tracker: str            # tracker config — "botsort.yaml" | "bytetrack.yaml" | custom path
    heartbeat: HeartbeatConfig
    health_file: HealthFileConfig
    buffer: BufferConfig


def parse(raw: dict) -> DeviceConfig:
    hb  = raw.get("heartbeat", {})
    hf  = raw.get("health_file", {})
    buf = raw.get("buffer", {})

    return DeviceConfig(
        id=raw["id"],
        name=raw.get("name", raw["id"]),
        location=raw.get("location", ""),
        environment=raw.get("environment", "production"),
        fps_target=raw.get("fps_target", 5),
        max_cameras=raw.get("max_cameras", 4),
        log_level=raw.get("log_level", "INFO"),
        tracker=raw.get("tracker", "botsort.yaml"),
        heartbeat=HeartbeatConfig(
            enabled=hb.get("enabled", True),
            interval_seconds=hb.get("interval_seconds", 60),
            table=hb.get("table", "nodes"),
        ),
        health_file=HealthFileConfig(
            enabled=hf.get("enabled", True),
            path=hf.get("path", "/tmp/visionengine-edge-health.json"),
            interval_seconds=hf.get("interval_seconds", 30),
        ),
        buffer=BufferConfig(
            path=buf.get("path", "./data/buffer.db"),
            max_size_mb=buf.get("max_size_mb", 200),
            retry_interval_seconds=buf.get("retry_interval_seconds", 10),
            delete_after_hours=buf.get("delete_after_hours", 24),
        ),
    )
