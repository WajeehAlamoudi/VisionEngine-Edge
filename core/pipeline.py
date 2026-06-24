from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone

import cv2

from core.buffer import Buffer
from core.config import CameraConfig
from core.ingest import IngestWorker
from core.model import InferenceResult, ModelRunner
from core.notifier import Notifier
from core.rules import Detection, RulesEngine
from core.zone import assign_zone

log = logging.getLogger(__name__)


class CameraPipeline:
    """
    Per-camera processing loop.

    Frame flow per inference cycle:
      capture → fps throttle → inference (thread pool)
        → zone assignment → rules evaluation → notify
        → route to table → buffer write
        → (every summary_interval_seconds) → summary flush
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
            collector=None,  # optional Collector — imported lazily to avoid circular import
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
        self._stop = asyncio.Event()
        self._rows_since_trigger = 0

        # summary accumulator: (zone, class_name) → {count, total_conf}
        self._summary: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"count": 0, "total_conf": 0.0}
        )

        # health stats — read by health.py
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
            "camera '%s': starting  source=%s  fps_target=%d  model=%s",
            self._cam.id, self._cam.source, self._cam.fps_target, self._cam.model_id,
        )

        cap = await loop.run_in_executor(None, cv2.VideoCapture, self._cam.source)
        if not cap.isOpened():
            self.last_error = f"failed to open source: {self._cam.source}"
            log.error("camera '%s': %s", self._cam.id, self.last_error)
            return

        try:
            while not self._stop.is_set():
                ret, frame = await loop.run_in_executor(None, cap.read)

                if not ret:
                    self.last_error = "frame read failed"
                    log.warning(
                        "camera '%s': failed to read frame — retrying in 2s", self._cam.id
                    )
                    await asyncio.sleep(2.0)
                    continue

                now = time.time()

                # fps throttle — skip frames between inference cycles
                if now - last_inference < frame_interval:
                    await asyncio.sleep(0)  # yield to other camera coroutines
                    continue

                last_inference = now
                self.frames_processed += 1

                # inference runs in thread pool — OpenCV + YOLO are synchronous
                try:
                    results = await loop.run_in_executor(
                        None, self._runner.predict, frame, self._cam.classes
                    )
                except Exception as exc:
                    self.last_error = str(exc)
                    log.error("camera '%s': inference error: %s", self._cam.id, exc)
                    continue

                ts = _utcnow()
                await self._process(results, ts)

                if self._collector:
                    await self._collector.on_frame(self._cam.id, frame, results, ts)

                # periodic summary flush
                if now - last_summary >= self._cam.summary_interval_seconds:
                    await self._flush_summary(_utcnow())
                    last_summary = now

        finally:
            await loop.run_in_executor(None, cap.release)
            log.info("camera '%s': stopped", self._cam.id)

    # ── per-frame processing ──────────────────────────────────────────────────

    async def _process(self, results, ts: str) -> None:
        rows: list[dict] = []

        for inf in results:
            x1, y1, x2, y2 = inf.bbox

            # zone assignment:
            # persons → bottom-center (where feet touch the ground)
            # everything else → bounding-box center
            cx = (x1 + x2) / 2
            cy = y2 if inf.class_name == "person" else (y1 + y2) / 2
            zone = assign_zone(cx, cy, self._cam.zones)

            det = Detection(
                camera_id=self._cam.id,
                camera_name=self._cam.name,
                class_name=inf.class_name,
                confidence=inf.confidence,
                bbox=inf.bbox,
                zone=zone,
                ts=ts,
            )

            self.detections_total += 1
            self.last_error = None

            # accumulate for summary
            acc = self._summary[(zone, inf.class_name)]
            acc["count"] += 1
            acc["total_conf"] += inf.confidence

            # route raw detection to target table
            table = self._route_table(inf.class_name)
            if table:
                rows.append({"table": table, "row": _detection_row(det)})

            # rules → alerts → notify
            matches = self._rules.evaluate(det)
            if matches:
                await self._notifier.notify(matches)
                for match in matches:
                    if match.rule.alerts_table:
                        rows.append({
                            "table": match.rule.alerts_table,
                            "row": _alert_row(match),
                        })

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
                    "camera_id": self._cam.id,
                    "zone": zone,
                    "class": class_name,
                    "count": acc["count"],
                    "avg_confidence": round(acc["total_conf"] / acc["count"], 4),
                    "ts": ts,
                },
            }
            for (zone, class_name), acc in self._summary.items()
            if acc["count"] > 0
        ]
        self._summary.clear()

        if rows:
            await self._buffer.write(rows)

    # ── routing ───────────────────────────────────────────────────────────────

    def _route_table(self, class_name: str) -> str | None:
        if self._cam.raw_table:
            return self._cam.raw_table

        # routing: first match wins, [] = all (universal convention)
        for entry in self._cam.routing:
            if not entry.classes or class_name in entry.classes:
                return entry.raw_table

        log.debug(
            "camera '%s': no routing entry for class '%s' — detection not stored",
            self._cam.id, class_name,
        )
        return None


# ── row builders ──────────────────────────────────────────────────────────────

def _detection_row(det: Detection) -> dict:
    return {
        "camera_id": det.camera_id,
        "class": det.class_name,
        "confidence": round(det.confidence, 4),
        "bbox_x1": int(det.bbox[0]),
        "bbox_y1": int(det.bbox[1]),
        "bbox_x2": int(det.bbox[2]),
        "bbox_y2": int(det.bbox[3]),
        "zone": det.zone,
        "ts": det.ts,
    }


def _alert_row(match) -> dict:
    det = match.detection
    return {
        "rule_name": match.rule.name,
        "class": det.class_name,
        "camera_id": det.camera_id,
        "zone": det.zone,
        "confidence": round(det.confidence, 4),
        "severity": match.rule.severity,
        "message": match.message,
        "ts": det.ts,
    }


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
