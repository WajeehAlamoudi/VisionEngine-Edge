from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RuleConfig:
    name: str
    class_name: str                  # mapped from YAML key 'class' ('class' is a Python keyword)
    cameras: list[str]               # [] = all cameras (universal convention)
    zones: list[str]                 # [] = all zones (universal convention)
    min_confidence: float | None
    cooldown_seconds: int
    severity: str                    # critical | warning | info
    notifications_table: str | None  # table where notification rows are written
    notify: bool                     # false = filter-only rule (no webhook, no notification row)
    message: str                     # supports placeholders: {class} {zone} {camera} {confidence}
    enabled: bool


def parse(raw: dict) -> RuleConfig:
    return RuleConfig(
        name=raw["name"],
        class_name=raw["class"],
        cameras=raw.get("cameras", []),
        zones=raw.get("zones", []),
        min_confidence=raw.get("min_confidence"),
        cooldown_seconds=raw.get("cooldown_seconds", 0),
        severity=raw.get("severity", "info"),
        notifications_table=raw.get("notifications_table"),
        notify=raw.get("notify", True),   # default True → existing rules unchanged
        message=raw.get("message", ""),
        enabled=raw.get("enabled", True),
    )
