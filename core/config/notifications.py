from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .rule import SEVERITIES
from .strict import ALL, ID_PATTERN, Reader, describe

_FILE = "notifications.yaml"

_TOP_KEYS = ("log", "webhooks")
_LOG_KEYS = ("enabled",)
_WEBHOOK_KEYS = ("name", "url", "enabled", "timeout_seconds", "rules", "severities")


@dataclass
class LogChannelConfig:
    enabled: bool


@dataclass
class WebhookConfig:
    name: str
    url: str
    enabled: bool
    timeout_seconds: int
    # [] = every rule / every severity, resolved from "*" at parse time.
    # Both filters must pass for a webhook to fire on a given match.
    rules: list[str]
    severities: list[str]

    def accepts(self, rule_name: str, severity: str) -> bool:
        """Whether this webhook should fire for a match from the given rule."""
        if self.rules and rule_name not in self.rules:
            return False
        if self.severities and severity not in self.severities:
            return False
        return True


@dataclass
class NotificationsConfig:
    log: LogChannelConfig
    webhooks: list[WebhookConfig]


def _parse_webhook(r: Reader) -> WebhookConfig:
    r.reject_unknown(*_WEBHOOK_KEYS)

    name = r.identifier(
        "name", ID_PATTERN,
        "letters, digits, hyphens, and underscores only (identifies this webhook in logs)",
    )
    url = r.url("url")
    enabled = r.boolean("enabled")
    timeout_seconds = r.integer("timeout_seconds", minimum=1)

    rules = r.string_list_or_all("rules")
    severities = r.string_list_or_all("severities")

    if severities is not ALL:
        for i, severity in enumerate(severities):
            if severity not in SEVERITIES:
                r.error(
                    f"{r.path_of('severities')}[{i}]",
                    f"expected one of {' | '.join(SEVERITIES)}, got '{severity}'",
                )

    return WebhookConfig(
        name=name,
        url=url,
        enabled=enabled,
        timeout_seconds=timeout_seconds,
        rules=[] if rules is ALL else list(rules),
        severities=[] if severities is ALL else list(severities),
    )


def parse(raw: Any) -> NotificationsConfig:
    """
    Parse and validate notifications.yaml.

    An empty `webhooks:` list is allowed here, unlike the empty lists rejected
    elsewhere. Everywhere else an empty list is a filter that would be
    ambiguous with "all", or a collection whose emptiness breaks the device.
    Zero webhooks is neither: it is a working configuration that delivers
    alerts to the log only.

    Whether a webhook may name a rule that does not exist is checked in
    _validate, where rules.yaml is in scope.
    """
    r = Reader(_FILE, raw)
    r.reject_unknown(*_TOP_KEYS)

    log_r = r.section("log")
    log_r.reject_unknown(*_LOG_KEYS)
    log_cfg = LogChannelConfig(enabled=log_r.boolean("enabled"))

    raw_webhooks = r.raw.get("webhooks", None)
    if "webhooks" not in r.raw:
        r.error("webhooks", "required key is missing (use an empty list for no webhooks)")
        raw_webhooks = []
    elif not isinstance(raw_webhooks, list):
        r.error("webhooks", f"expected a list of webhooks, got {describe(raw_webhooks)}")
        raw_webhooks = []

    webhooks: list[WebhookConfig] = []
    seen: set[str] = set()

    for i, item in enumerate(raw_webhooks):
        path = f"webhooks[{i}]"
        webhook = _parse_webhook(r.child(path, item))

        if webhook.name:
            if webhook.name in seen:
                r.error(f"{path}.name", f"duplicate webhook name '{webhook.name}'")
            seen.add(webhook.name)

        webhooks.append(webhook)

    r.raise_if_errors()
    return NotificationsConfig(log=log_cfg, webhooks=webhooks)
