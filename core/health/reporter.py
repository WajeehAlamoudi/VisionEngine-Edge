from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from core.buffer import Buffer
from core.config import AppConfig
from .rows import _heartbeat_row, _utcnow, _write_json

log = logging.getLogger(__name__)


class HealthReporter:
    """
    Two independent background loops:

    health_file  — writes a JSON snapshot to local disk every health_file.interval_seconds.
                   No network call. Readable via SSH or a local watchdog script.

    heartbeat    — pushes one row to the nodes table (via the buffer) every
                   heartbeat.interval_seconds. Appears on the VisionEngine dashboard
                   as the device's online/offline status.

    Both loops are independent. A heartbeat failure does not affect the health file
    and vice versa.
    """

    def __init__(self, cfg: AppConfig, pipelines: list, buffer: Buffer) -> None:
        self._cfg = cfg
        self._pipelines = pipelines
        self._buffer = buffer
        self._started_at = time.time()
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        if self._cfg.device.health_file.enabled:
            self._tasks.append(
                asyncio.create_task(self._health_file_loop(), name="health-file")
            )
        if self._cfg.device.heartbeat.enabled:
            self._tasks.append(
                asyncio.create_task(self._heartbeat_loop(), name="heartbeat")
            )

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ── loops ─────────────────────────────────────────────────────────────────

    async def _health_file_loop(self) -> None:
        cfg = self._cfg.device.health_file
        path = Path(cfg.path)
        loop = asyncio.get_event_loop()

        while not self._stop.is_set():
            try:
                health = await self._build_health()
                await loop.run_in_executor(None, _write_json, path, health)
            except Exception as exc:
                log.warning("health file: write failed: %s", exc)

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=cfg.interval_seconds)
            except asyncio.TimeoutError:
                pass

    async def _heartbeat_loop(self) -> None:
        cfg = self._cfg.device.heartbeat

        while not self._stop.is_set():
            try:
                health = await self._build_health()
                await self._buffer.write([{
                    "table": cfg.table,
                    "row":   _heartbeat_row(health),
                }])
                log.debug("heartbeat: pushed to '%s'", cfg.table)
            except Exception as exc:
                log.warning("heartbeat: failed: %s", exc)

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=cfg.interval_seconds)
            except asyncio.TimeoutError:
                pass

    # ── health snapshot ───────────────────────────────────────────────────────

    async def _build_health(self) -> dict:
        buf_stats = await self._buffer.stats()
        uptime = int(time.time() - self._started_at)
        ts = _utcnow()

        cameras = []
        cameras_active = 0
        cameras_error = 0

        for p in self._pipelines:
            has_error = p.last_error is not None
            cameras.append({
                "id":               p._cam.id,
                "name":             p._cam.name,
                "status":           "error" if has_error else "ok",
                "detections_total": p.detections_total,
                "frames_processed": p.frames_processed,
                "last_error":       p.last_error,
            })
            if has_error:
                cameras_error += 1
            else:
                cameras_active += 1

        if cameras_error == 0:
            status = "ok"
        elif cameras_active == 0:
            status = "error"
        else:
            status = "degraded"

        return {
            "device_id":      self._cfg.device.id,
            "name":           self._cfg.device.name,
            "location":       self._cfg.device.location,
            "status":         status,
            "uptime_seconds": uptime,
            "cameras":        cameras,
            "cameras_active": cameras_active,
            "cameras_error":  cameras_error,
            "buffer":         buf_stats,
            "ts":             ts,
        }
