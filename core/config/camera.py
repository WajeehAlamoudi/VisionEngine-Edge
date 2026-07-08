from __future__ import annotations

from dataclasses import dataclass


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
    fps_target: int
    model_id: str
    classes: list[str]
    confidence_threshold: float | None
    raw_table: str | None       # single-table mode
    routing: list[RoutingEntry] # class-routing mode (mutually exclusive with raw_table)
    zones: list[Zone]           # [] = full frame


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
        zones=[
            Zone(name=z["name"], polygon=z["polygon"])
            for z in raw.get("zones", [])
        ],
    )
