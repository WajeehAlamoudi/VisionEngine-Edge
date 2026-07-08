from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

import httpx

from core.buffer import Buffer, BufferedRow
from core.config import AppConfig

log = logging.getLogger(__name__)


class IngestWorker:
    """
    Background worker that flushes buffered detection rows to the VisionEngine API.

    Flush triggers (whichever comes first):
      - flush_interval_seconds has elapsed
      - pipeline calls trigger() after writing batch_size rows

    On failure:
      - rows are marked back to pending and retried on the next flush
      - consecutive failures increase the wait time (exponential backoff)
      - after max_consecutive_failures, backoff is capped at 5× retry_interval_seconds

    One API call is made per target table per flush, so routing cameras
    (person_detections, vehicle_detections, etc.) each get their own request.
    """

    def __init__(self, cfg: AppConfig, buffer: Buffer) -> None:
        self._url = f"{cfg.api.url}/branches/ingest"
        self._headers = {"X-API-Key": cfg.api.key}
        self._batch_size = cfg.api.ingest.batch_size
        self._flush_interval = cfg.api.ingest.flush_interval_seconds
        self._timeout = cfg.api.request.timeout_seconds
        self._max_failures = cfg.api.request.max_consecutive_failures
        self._retry_interval = cfg.api.buffer.retry_interval_seconds
        self._buffer = buffer
        self._trigger = asyncio.Event()
        self._stop = asyncio.Event()
        self._consecutive_failures = 0
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="ingest-worker")
        log.info("ingest: worker started (flush every %ds, batch=%d)",
                 self._flush_interval, self._batch_size)

    async def stop(self) -> None:
        """Signal the loop to stop and wait for the current flush to finish."""
        self._stop.set()
        self._trigger.set()
        if self._task:
            await self._task
        log.info("ingest: worker stopped")

    def trigger(self) -> None:
        """Request an early flush. Called by the pipeline after writing batch_size rows."""
        self._trigger.set()

    # ── internal loop ─────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            while not self._stop.is_set():
                delay = self._backoff_delay()
                try:
                    await asyncio.wait_for(self._trigger.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                self._trigger.clear()

                if self._stop.is_set():
                    break

                await self._flush(client)
                await self._buffer.purge_old_sent()

    async def _flush(self, client: httpx.AsyncClient) -> None:
        rows = await self._buffer.get_pending(limit=self._batch_size)
        if not rows:
            return

        by_table: dict[str, list[BufferedRow]] = defaultdict(list)
        for row in rows:
            by_table[row.table].append(row)

        for table, batch in by_table.items():
            success = await self._push(client, table, batch)
            if success:
                await self._buffer.mark_sent([r.id for r in batch])
                self._consecutive_failures = 0
                log.info("ingest: flushed %d row(s) → '%s'", len(batch), table)
            else:
                await self._buffer.mark_failed([r.id for r in batch])
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_failures:
                    log.warning(
                        "ingest: %d consecutive failures — next retry in %.0fs",
                        self._consecutive_failures,
                        self._backoff_delay(),
                    )

    async def _push(
            self, client: httpx.AsyncClient, table: str, batch: list[BufferedRow]
    ) -> bool:
        payload = {"table": table, "rows": [r.row for r in batch]}
        try:
            resp = await client.post(self._url, json=payload, headers=self._headers)
            if resp.status_code in (200, 201):
                return True
            log.warning(
                "ingest: API %d for table '%s' — %s",
                resp.status_code, table, resp.text[:200],
            )
            return False
        except httpx.TimeoutException:
            log.warning("ingest: timeout pushing to table '%s'", table)
            return False
        except httpx.RequestError as exc:
            log.warning("ingest: connection error for table '%s': %s", table, exc)
            return False

    def _backoff_delay(self) -> float:
        if self._consecutive_failures == 0:
            return float(self._flush_interval)
        factor = min(2 ** (self._consecutive_failures - 1), 5)
        return float(self._retry_interval * factor)
