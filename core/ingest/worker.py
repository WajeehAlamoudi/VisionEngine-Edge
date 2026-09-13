from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

import httpx

from core.buffer import Buffer, BufferedRow
from core.config import AppConfig

log = logging.getLogger(__name__)

# How long stop() waits for a flush that is already in flight before cancelling
# it. One request per table, so this is a few timeouts, not one.
_LOOP_EXIT_SECONDS = 10

# Ceiling on the shutdown drain. Reached only when the backend is unreachable —
# which is usually why rows are still buffered — so it has to be short enough
# that Ctrl+C still feels like a stop.
_SHUTDOWN_FLUSH_SECONDS = 15


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
        # Owned here rather than by _loop, so the shutdown drain can keep using
        # it after the loop has ended — including when the loop had to be
        # cancelled, which would have closed a client scoped to it.
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=self._timeout)
        self._task = asyncio.create_task(self._loop(), name="ingest-worker")
        log.info("ingest: worker started (flush every %ds, batch=%d)",
                 self._flush_interval, self._batch_size)

    async def stop(self) -> None:
        """
        End the loop, then push what is still buffered before returning.

        The drain is the point of this method. Buffered rows survive a restart
        either way — SQLite has them — but without it they sit on the device
        until the agent comes back, which is minutes after a reboot and open
        ended when the restart is by hand. Sending them now costs a few seconds
        of shutdown.

        Both waits are bounded, because an unreachable backend is the usual
        reason rows are still here and Ctrl+C has to stay a stop. Whatever does
        not go out stays pending for the next start, exactly as it would have.
        """
        self._stop.set()
        self._trigger.set()

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=_LOOP_EXIT_SECONDS)
            except asyncio.TimeoutError:
                log.warning("ingest: flush in flight did not finish in %ds — cancelling",
                            _LOOP_EXIT_SECONDS)
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)

        try:
            await asyncio.wait_for(self._drain(), timeout=_SHUTDOWN_FLUSH_SECONDS)
        except asyncio.TimeoutError:
            log.warning(
                "ingest: could not empty the buffer within %ds — the rest stays "
                "buffered and goes out on the next start", _SHUTDOWN_FLUSH_SECONDS,
            )
        except Exception as exc:
            log.warning("ingest: shutdown flush failed: %s — rows stay buffered", exc)
        finally:
            if self._client:
                await self._client.aclose()
                self._client = None

        log.info("ingest: worker stopped")

    async def _drain(self) -> None:
        """
        Flush batch after batch until the buffer is empty or a push fails.

        Stopping on the first failure is deliberate. The backoff the running
        loop uses exists to keep retrying for hours; here there are seconds, and
        a backend that just rejected a batch will reject the next one too. Give
        up and leave them pending rather than spend the deadline proving it.
        """
        if self._client is None:      # stopped without ever having started
            return

        sent = 0
        while True:
            pushed = await self._flush(self._client)
            if pushed == 0:
                break
            sent += pushed

        pending = len(await self._buffer.get_pending(limit=1))
        if sent:
            log.info("ingest: flushed %d buffered row(s) before exit", sent)
        if pending:
            log.warning("ingest: rows remain buffered — they go out on the next start")

    def trigger(self) -> None:
        """Request an early flush. Called by the pipeline after writing batch_size rows."""
        self._trigger.set()

    # ── internal loop ─────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while not self._stop.is_set():
            delay = self._backoff_delay()
            try:
                await asyncio.wait_for(self._trigger.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            self._trigger.clear()

            # No flush here on the way out. stop() drains the buffer itself,
            # under its own deadline — flushing once more here would be a batch
            # short of empty and would run before that deadline started.
            if self._stop.is_set():
                break

            await self._flush(self._client)
            await self._buffer.purge_old_sent()

    async def _flush(self, client: httpx.AsyncClient) -> int:
        """
        Push one batch. Returns the number of rows accepted by the API.

        Zero means there was nothing pending or a push failed — the two cases
        the shutdown drain treats the same way, since neither is worth another
        attempt inside the seconds it has.
        """
        rows = await self._buffer.get_pending(limit=self._batch_size)
        if not rows:
            return 0

        by_table: dict[str, list[BufferedRow]] = defaultdict(list)
        for row in rows:
            by_table[row.table].append(row)

        sent = 0
        for table, batch in by_table.items():
            success = await self._push(client, table, batch)
            if success:
                await self._buffer.mark_sent([r.id for r in batch])
                self._consecutive_failures = 0
                sent += len(batch)
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
        return sent

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
