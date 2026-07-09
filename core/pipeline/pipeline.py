from __future__ import annotations

import asyncio
import logging
import os
import time

import cv2

from core.buffer import Buffer
from core.collector import Collector
from core.config import CameraConfig
from core.ingest import IngestWorker
from core.model import ModelRunner
from core.notifier import Notifier
from core.rules import RulesEngine
from .enricher import enrich
from .rows import _utcnow, detection_row, notification_row

log = logging.getLogger(__name__)


class CameraPipeline:
    """
    Per-camera processing loop.

    Frame flow per inference cycle:
      capture → fps throttle → inference (thread pool)
        → enrich (DetectionEvent) → rules filter+tag
        → detection row → notification row (if rule fires)
        → buffer write → ingest trigger
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
        self._cam      = cam
        self._runner   = runner
        self._buffer   = buffer
        self._rules    = rules
        self._notifier = notifier
        self._ingest   = ingest
        self._device_id = device_id
        self._batch_size = batch_size
        self._collector  = collector
        self._stop = asyncio.Event()
        self._rows_since_trigger = 0

        self._raw_table   = cam.raw_table
        self._routing     = cam.routing

        # health stats — read by HealthReporter
        self.detections_total = 0
        self.frames_processed = 0
        self.last_error: str | None = None

    def stop(self) -> None:
        self._stop.set()

    @staticmethod
    def _open_cap(source: str) -> tuple[cv2.VideoCapture, int, int]:
        # TCP + discard-corrupt handles H.264, H.265, H.264+ (Hikvision/Dahua non-standard SPS)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;+discardcorrupt+genpts"
        )
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            return cap, 0, 0
        # Decode until we get a valid frame — grab() alone won't recover a
        # non-standard H.264+ stream with bad SPS headers.
        w, h = 0, 0
        for _ in range(120):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                break
        if w == 0:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap, w, h

    async def run(self) -> None:
        loop = asyncio.get_event_loop()
        frame_interval = 1.0 / self._cam.fps_target

        log.info(
            "camera '%s': starting  source=%s  fps_target=%d  model=%s",
            self._cam.id, self._cam.source, self._cam.fps_target, self._cam.model_id,
        )

        cap, frame_w, frame_h = await loop.run_in_executor(None, self._open_cap, self._cam.source)
        if not cap.isOpened():
            self.last_error = f"failed to open source: {self._cam.source}"
            log.error("camera '%s': %s", self._cam.id, self.last_error)
            return

        log.info("camera '%s': stream ready  %dx%d", self._cam.id, frame_w, frame_h)
        last_inference = 0.0

        try:
            while not self._stop.is_set():
                cap_ts = _utcnow()
                ret, frame = await loop.run_in_executor(None, cap.read)

                if not ret:
                    self.last_error = "frame read failed"
                    log.warning("camera '%s': failed to read frame — retrying in 2s", self._cam.id)
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

                log.debug(
                    "camera '%s': frame %d — %d detection(s): %s",
                    self._cam.id, self.frames_processed, len(results),
                    [(r.class_name, round(r.confidence, 2)) for r in results] if results else "none",
                )
                await self._process(results, cap_ts, frame_w, frame_h)

                if self._collector:
                    await self._collector.on_frame(self._cam.id, frame, results, cap_ts)

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
        detection_rows:    list[dict] = []
        notification_rows: list[dict] = []

        for inf in results:
            # 1. Enrich — raw InferenceResult → full DetectionEvent
            event = enrich(inf, self._cam, frame_w, frame_h, capture_ts)

            # 2. Rules — filter irrelevant detections, tag relevant ones
            matches = self._rules.filter_and_tag(event)
            if matches is None:
                continue    # no rule matched → discard

            # 3. Detection row
            raw_table = self._route_table(event.class_name)
            if raw_table:
                detection_rows.append({"table": raw_table, "row": detection_row(event)})

            # 4. Notification rows
            if matches:
                await self._notifier.notify(matches)
                for match in matches:
                    if match.rule.notifications_table:
                        notification_rows.append({
                            "table": match.rule.notifications_table,
                            "row":   notification_row(match),
                        })

            self.detections_total += 1
            self.last_error = None

        rows = [*detection_rows, *notification_rows]
        if not rows:
            return

        await self._buffer.write(rows)
        self._rows_since_trigger += len(rows)
        if self._rows_since_trigger >= self._batch_size:
            self._ingest.trigger()
            self._rows_since_trigger = 0

    def _route_table(self, class_name: str) -> str | None:
        if self._raw_table:
            return self._raw_table
        for entry in self._routing:
            if not entry.classes or class_name in entry.classes:
                return entry.raw_table
        return None
