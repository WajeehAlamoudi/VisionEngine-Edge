from __future__ import annotations

import asyncio
import logging

import httpx

from core.config import AppConfig, WebhookConfig
from core.rules import RuleMatch
from .payload import _build_payload

log = logging.getLogger(__name__)


class Notifier:
    """
    Delivers alerts to configured channels when a rule fires.

    Log channel  — synchronous, fires immediately inside notify().
    Webhooks     — launched as background asyncio tasks so the pipeline
                   is not blocked waiting for HTTP responses.

    All enabled webhooks fire in parallel for each match.
    One webhook timing out or failing does not affect the others.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._log_enabled = cfg.notifications.log.enabled
        self._webhooks = cfg.enabled_webhooks
        self._device_id = cfg.device.id
        self._branch_id = cfg.api.branch_id
        self._api_key = cfg.api.key
        self._client: httpx.AsyncClient | None = None
        self._pending: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._client = httpx.AsyncClient()

    async def stop(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
        if self._client:
            await self._client.aclose()

    async def notify(self, matches: list[RuleMatch]) -> None:
        for match in matches:
            self._log(match)

            if not self._webhooks:
                continue

            payload = _build_payload(match, self._device_id, self._branch_id)
            for webhook in self._webhooks:
                task = asyncio.create_task(
                    self._send(webhook, payload),
                    name=f"webhook-{webhook.name}",
                )
                self._pending.add(task)
                task.add_done_callback(self._pending.discard)

    def _log(self, match: RuleMatch) -> None:
        if not self._log_enabled:
            return
        det = match.detection
        rule = match.rule
        log.info(
            "ALERT [%s] rule=%s  %s in %s  camera=%s  conf=%.2f  track_id=%s",
            rule.severity.upper(), rule.name,
            det.class_name, det.zone, det.camera_id, det.confidence, det.track_id,
        )

    async def _send(self, webhook: WebhookConfig, payload: dict) -> None:
        try:
            resp = await self._client.post(
                webhook.url,
                json=payload,
                headers={"X-API-Key": self._api_key},
                timeout=webhook.timeout_seconds,
            )
            if resp.status_code >= 400:
                log.warning(
                    "notifier: webhook '%s' returned %d",
                    webhook.name, resp.status_code,
                )
        except httpx.TimeoutException:
            log.warning("notifier: webhook '%s' timed out after %ds",
                        webhook.name, webhook.timeout_seconds)
        except httpx.RequestError as exc:
            log.warning("notifier: webhook '%s' error: %s", webhook.name, exc)
