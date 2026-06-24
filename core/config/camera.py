from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Zone:
    name: str
    polygon: list[list[int]]    # [[x, y], ...] pixel coordinates


@dataclass
class RoutingEntry:
    classes: list[str]          # [] = all (universal convention)
    raw_table: str


@dataclass
class CameraConfig:
    id: str
    name: str
    source: str | int           # int = USB index, str = RTSP URL or file path
    enabled: bool
    fps_target: int             # resolved: camera override or device default
    model_id: str
    classes: list[str]          # resolved at validation: camera filter or full model list
    confidence_threshold: float | None  # resolved at validation: override or model floor
    raw_table: str | None       # single-table mode
    routing: list[RoutingEntry] # class-routing mode (mutually exclusive with raw_table)
    summary_table: str | None
    summary_interval_seconds: int
    zones: list[Zone]           # [] = full frame (universal convention)


def parse(raw: dict, device_fps: int) -> CameraConfig:
    return CameraConfig(
        id=raw["id"],
        name=raw.get("name", raw["id"]),
        source=raw["source"],
        enabled=raw.get("enabled", True),
        fps_target=raw.get("fps_target", device_fps),
        model_id=raw.get("model_id", ""),
        classes=raw.get("classes", []),
        confidence_threshold=raw.get("confidence_threshold"),
        raw_table=raw.get("raw_table"),
        routing=[
            RoutingEntry(
                classes=r.get("classes", []),
                raw_table=r["raw_table"],
            )
            for r in raw.get("routing", [])
        ],
        summary_table=raw.get("summary_table"),
        summary_interval_seconds=raw.get("summary_interval_seconds", 60),
        zones=[
            Zone(name=z["name"], polygon=z["polygon"])
            for z in raw.get("zones", [])
        ],
    )
