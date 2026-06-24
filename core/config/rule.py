from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RuleConfig:
    name: str
    class_name: str             # mapped from YAML key 'class' ('class' is a Python keyword)
    zones: list[str]            # [] = all zones (universal convention)
    min_confidence: float | None
    cooldown_seconds: int
    severity: str               # critical | warning | info
    alerts_table: str | None
    message: str                # supports placeholders: {class} {zone} {camera} {confidence}
    enabled: bool


def parse(raw: dict) -> RuleConfig:
    return RuleConfig(
        name=raw["name"],
        class_name=raw["class"],
        zones=raw.get("zones", []),
        min_confidence=raw.get("min_confidence"),
        cooldown_seconds=raw.get("cooldown_seconds", 0),
        severity=raw.get("severity", "info"),
        alerts_table=raw.get("alerts_table"),
        message=raw.get("message", ""),
        enabled=raw.get("enabled", True),
    )
