from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict

import cv2

from core.analytics import AnalyticsEngine
from core.buffer import Buffer
from core.config import CameraConfig
from core.ingest import IngestWorker
from core.model import ModelRunner
from core.notifier import Notifier
from core.rules import RulesEngine
from .enricher import enrich
from .rows import _utcnow, notification_row

log = logging.getLogger(__name__)


class CameraPipeline:
    """
    Per-camera processing loop.

    Frame flow per inference cycle:
      capture → fps throttle → inference / tracking (thread pool)
        → enrich (DetectionEvent) → rules filter+tag
        → analytics engine (raw store, dwell, occupancy, trajectory)
        → alerts → buffer write
        → (every summary_interval_seconds) → summary flush + analytics periodic flush
    """

    def __init__(
            self,
            cam: CameraConfig,
            runner: ModelRunner,
            buffer: Buffer,
            rules: RulesEngine,
            notifier: Notifier,
            ingest: IngestWorker,
            device_id: str,
            batch_size: int,
            collector=None,
    ) -> None:
        self._cam = cam
        self._runner = runner
        self._buffer = buffer
        self._rules = rules
        self._notifier = notifier
        self._ingest = ingest
        self._device_id = device_id
        self._batch_size = batch_size
        self._collector = collector
        self._analytics = AnalyticsEngine(cam)
        self._stop = asyncio.Event()
        self._rows_since_trigger = 0

        # summary accumulator (ALL detections, pre-filter) → summary_table
        self._summary: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"count": 0, "total_conf": 0.0}
        )

        # health stats — read by HealthReporter
        self.detections_total = 0
        self.frames_processed = 0
        self.last_error: str | None = None

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        loop = asyncio.get_event_loop()
        frame_interval = 1.0 / self._cam.fps_target
        last_inference = 0.0
        last_summary = time.time()

        log.info(
            "camera '%s': starting  source=%s  fps_target=%d  model=%s  tracker=%s",
            self._cam.id, self._cam.source, self._cam.fps_target,
            self._cam.model_id, self._cam.analytics.needs_tracker,
        )

        cap = await loop.run_in_executor(None, cv2.VideoCapture, self._cam.source)
        if not cap.isOpened():
            self.last_error = f"failed to open source: {self._cam.source}"
            log.error("camera '%s': %s", self._cam.id, self.last_error)
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            while not self._stop.is_set():
                # grab timestamp at read time — not after inference
                cap_ts = _utcnow()
                ret, frame = await loop.run_in_executor(None, cap.read)

                if not ret:
                    self.last_error = "frame read failed"
                    log.warning(
                        "camera '%s': failed to read frame — retrying in 2s", self._cam.id
                    )
                    await asyncio.sleep(2.0)
                    continue

                now = time.time()
                if now - last_inference < frame_interval:
                    await asyncio.sleep(0)
                    continue

                last_inference = now
                self.frames_processed += 1

                try:
                    results = await loop.run_in_executor(
                        None, self._runner.run, frame, self._cam.classes
                    )
                except Exception as exc:
                    self.last_error = str(exc)
                    log.error("camera '%s': inference error: %s", self._cam.id, exc)
                    continue

                await self._process(results, cap_ts, frame_w, frame_h)

                if self._collector:
                    await self._collector.on_frame(self._cam.id, frame, results, cap_ts)

                if now - last_summary >= self._cam.summary_interval_seconds:
                    flush_ts = _utcnow()
                    await self._flush_summary(flush_ts)
                    analytics_rows = self._analytics.flush_periodic(flush_ts)
                    if analytics_rows:
                        await self._buffer.write(analytics_rows)
                    last_summary = now

        finally:
            await loop.run_in_executor(None, cap.release)
            log.info("camera '%s': stopped", self._cam.id)

    # ── per-frame processing ──────────────────────────────────────────────────

    async def _process(
            self,
            results,
            capture_ts: str,
            frame_w: int,
            frame_h: int,
    ) -> None:
        rows: list[dict] = []

        for inf in results:
            # 1. Enrich — raw InferenceResult → full DetectionEvent
            event = enrich(inf, self._cam, frame_w, frame_h, capture_ts)

            # 2. Summary accumulator — all detections, before rules filter
            acc = self._summary[(event.zone, event.class_name)]
            acc["count"] += 1
            acc["total_conf"] += event.confidence

            # 3. Rules — filter irrelevant detections, tag relevant ones
            matches = self._rules.filter_and_tag(event)
            if matches is None:
                continue    # no rule matched → discard

            # 4. Analytics (raw store, dwell, occupancy, trajectory)
            rows.extend(self._analytics.process(event))

            # 5. Notifications
            if matches:
                await self._notifier.notify(matches)
                for match in matches:
                    if match.rule.notifications_table:
                        rows.append({
                            "table": match.rule.notifications_table,
                            "row":   notification_row(match),
                        })

            self.detections_total += 1
            self.last_error = None

        if not rows:
            return

        await self._buffer.write(rows)
        self._rows_since_trigger += len(rows)
        if self._rows_since_trigger >= self._batch_size:
            self._ingest.trigger()
            self._rows_since_trigger = 0

    async def _flush_summary(self, ts: str) -> None:
        if not self._cam.summary_table:
            self._summary.clear()
            return

        rows = [
            {
                "table": self._cam.summary_table,
                "row": {
                    "camera_id":      self._cam.id,
                    "zone":           zone,
                    "class":          class_name,
                    "count":          acc["count"],
                    "avg_confidence": round(acc["total_conf"] / acc["count"], 4),
                    "ts":             ts,
                },
            }
            for (zone, class_name), acc in self._summary.items()
            if acc["count"] > 0
        ]
        self._summary.clear()

        if rows:
            await self._buffer.write(rows)
