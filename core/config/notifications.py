from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LogChannelConfig:
    enabled: bool


@dataclass
class WebhookConfig:
    name: str
    url: str
    enabled: bool
    timeout_seconds: int


@dataclass
class NotificationsConfig:
    log: LogChannelConfig
    webhooks: list[WebhookConfig]


def parse(raw: dict) -> NotificationsConfig:
    log = raw.get("log", {})
    return NotificationsConfig(
        log=LogChannelConfig(enabled=log.get("enabled", True)),
        webhooks=[
            WebhookConfig(
                name=w["name"],
                url=w["url"],
                enabled=w.get("enabled", False),
                timeout_seconds=w.get("timeout_seconds", 5),
            )
            for w in raw.get("webhooks", [])
        ],
    )
