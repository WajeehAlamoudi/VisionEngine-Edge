from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from core.buffer import Buffer
from core.collector import Collector
from core.config import CameraConfig
from core.ingest import IngestWorker
from core.notifier import Notifier
from core.rules import RulesEngine
from core.model.detector import CameraRuntime, SourceUnavailable
from .enricher import enrich
from .rows import _utcnow, detection_row, notification_row

log = logging.getLogger(__name__)

# How often each camera logs its measured throughput, in seconds. Short enough
# to watch a device settle after start and to catch a camera degrading while it
# happens. Each camera reports independently, so this is one line per camera
# per interval.
_FPS_LOG_INTERVAL = 10.0

# Retry pacing after a read fails. Doubles per consecutive failure so a brief
# network blip costs a second and a camera that has gone away is retried once a
# minute instead of continuously.
_RETRY_BASE_SECONDS = 2.0
_RETRY_MAX_SECONDS = 60.0


class CameraPipeline:
    """
    Per-camera processing loop.

    Video is not handled here. A CameraRuntime opens the source, paces it to
    fps_target and returns detections; how it does that differs completely
    between runtimes and is none of this loop's business. What remains is the
    part that is identical for every camera on every runtime.

    Per cycle:
      camera runtime → detections (thread pool)
        → enrich (DetectionEvent) → rules filter+tag
        → detection row → notification row (if rule fires)
        → buffer write → ingest trigger
    """

    def __init__(
            self,
            cam: CameraConfig,
            camera_runtime: CameraRuntime,
            buffer: Buffer,
            rules: RulesEngine,
            notifier: Notifier,
            ingest: IngestWorker,
            device_id: str,
            batch_size: int,
            collector=None,
    ) -> None:
        self._cam = cam
        self._runtime = camera_runtime
        self._buffer = buffer
        self._rules = rules
        self._notifier = notifier
        self._ingest = ingest
        self._device_id = device_id
        self._batch_size = batch_size
        self._collector = collector
        self._stop = asyncio.Event()
        self._rows_since_trigger = 0

        self._raw_table = cam.raw_table
        self._routing = cam.routing

        # health stats — read by HealthReporter
        self.detections_total = 0
        self.frames_processed = 0
        self.last_error: str | None = None

        # throughput reporting — measured over the last interval, not since
        # start, so the figure reflects current load rather than being dragged
        # down by model warmup
        self._last_report_at = 0.0
        self._frames_at_report = 0
        self._detections_at_report = 0

    def stop(self) -> None:
        """
        Ask the loop to finish, and unblock it so it can.

        Setting the event alone is not enough: the loop spends nearly all its
        time inside a blocking read() on its own thread, and cannot look at the
        event until that returns. A camera that has gone quiet would hold
        shutdown until its read timed out, which is seconds per camera.

        request_stop() is what releases it, and is deliberately not close():
        this runs on a different thread from the one inside read(), and
        releasing a source while a read is using it is a race. Each runtime
        decides what is safe to call concurrently, and a runtime whose read
        returns promptly on its own does nothing at all here. Closing stays
        with the loop, after it has left read().
        """
        self._stop.set()
        try:
            self._runtime.request_stop()
        except Exception:
            log.debug("camera '%s': error interrupting the source during stop",
                      self._cam.id, exc_info=True)

    async def run(self) -> None:
        loop = asyncio.get_event_loop()

        log.info(
            "camera '%s': starting  source=%s  fps_target=%d  model=%s",
            self._cam.id, self._cam.source, self._cam.fps_target, self._cam.model_id,
        )

        # A camera's own thread, not the default executor. read() blocks for as
        # long as the fps_target interval, so a camera on the shared pool would
        # hold one of its few threads continuously — with a handful of cameras
        # that starves the collector's file writes and the health reporter.
        # One thread each also keeps the promise that a stalled camera does not
        # affect the others.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"cam-{self._cam.id}")

        try:
            try:
                await loop.run_in_executor(executor, self._runtime.open)
            except Exception as exc:
                self.last_error = f"failed to open source: {exc}"
                log.error("camera '%s': %s", self._cam.id, self.last_error)
                # close() even though open() failed: it may have got far enough
                # to hold something — a GStreamer pipeline can reach PLAYING and
                # then time out waiting for video, and would otherwise keep the
                # connection and the decoder session until the process exits.
                await loop.run_in_executor(executor, self._runtime.close)
                return

            frame_w, frame_h = self._runtime.frame_size
            log.info("camera '%s': stream ready  %dx%d", self._cam.id, frame_w, frame_h)
            self._last_report_at = time.time()

            consecutive_errors = 0
            while not self._stop.is_set():
                try:
                    frame, results = await loop.run_in_executor(
                        executor, self._runtime.read)
                except SourceUnavailable as exc:
                    # stop() closes the source to release a blocked read, so
                    # this is the expected way a shutdown arrives here. Leave
                    # rather than announce a retry that will not happen.
                    if self._stop.is_set():
                        break
                    consecutive_errors += 1
                    self.last_error = str(exc)
                    delay = self._retry_delay(consecutive_errors)
                    log.warning("camera '%s': %s — retrying in %.0fs",
                                self._cam.id, exc, delay)
                    await asyncio.sleep(delay)
                    continue
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    # An inference failure that repeats is usually permanent —
                    # a dead pipeline returns instantly, so without a delay this
                    # loop would spin at full speed logging every iteration.
                    consecutive_errors += 1
                    self.last_error = str(exc)
                    delay = self._retry_delay(consecutive_errors)
                    log.error("camera '%s': inference error: %s — retrying in %.0fs",
                              self._cam.id, exc, delay)
                    await asyncio.sleep(delay)
                    continue

                # Stamped after the read returns, not before it. read() absorbs
                # the fps_target wait, so a timestamp taken beforehand would be
                # up to one interval earlier than the frame it describes.
                cap_ts = _utcnow()
                consecutive_errors = 0
                self.frames_processed += 1

                log.debug(
                    "camera '%s': frame %d — %d detection(s): %s",
                    self._cam.id, self.frames_processed, len(results),
                    [(r.class_name, round(r.confidence, 2)) for r in results] if results else "none",
                )
                await self._process(results, cap_ts, frame_w, frame_h)
                self._report_throughput(time.time())

                # A runtime that keeps frames on the GPU returns None; there is
                # nothing for the collector to save in that case.
                if self._collector and frame is not None:
                    await self._collector.on_frame(self._cam.id, frame, results, cap_ts)

            await loop.run_in_executor(executor, self._runtime.close)
        finally:
            executor.shutdown(wait=False)
            log.info("camera '%s': stopped", self._cam.id)

    @staticmethod
    def _retry_delay(consecutive_errors: int) -> float:
        """
        Back off as failures repeat, up to a ceiling.

        A source that drops briefly should be retried quickly; one that is gone
        should not be retried a thousand times a minute. The ceiling keeps a
        camera that recovers hours later from being ignored.
        """
        return min(_RETRY_BASE_SECONDS * (2 ** (consecutive_errors - 1)), _RETRY_MAX_SECONDS)

    # ── per-frame processing ──────────────────────────────────────────────────

    async def _process(
            self,
            results,
            capture_ts: str,
            frame_w: int,
            frame_h: int,
    ) -> None:
        detection_rows: list[dict] = []
        notification_rows: list[dict] = []

        for inf in results:
            # 1. Enrich — raw InferenceResult → full DetectionEvent
            event = enrich(inf, self._cam, frame_w, frame_h, capture_ts)

            # 2. Rules — filter irrelevant detections, tag relevant ones
            matches = self._rules.filter_and_tag(event)
            if matches is None:
                continue  # no rule matched → discard

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
                            "row": notification_row(match),
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

    def _report_throughput(self, now: float) -> None:
        """
        Log measured inference rate every _FPS_LOG_INTERVAL seconds.

        Rate is computed from the frames processed since the previous report,
        so it reflects current load. A cumulative average would stay depressed
        by model warmup long after the device had settled.
        """
        elapsed = now - self._last_report_at
        if elapsed < _FPS_LOG_INTERVAL:
            return

        frames = self.frames_processed - self._frames_at_report
        detections = self.detections_total - self._detections_at_report
        fps = frames / elapsed

        # fps_target is the ceiling this camera is throttled to; falling well
        # short of it means the device is saturated, not idle.
        log.info(
            "camera '%s': %.1f fps (target %d)  |  %d frames, %d detections in %.0fs",
            self._cam.id, fps, self._cam.fps_target, frames, detections, elapsed,
        )

        self._last_report_at = now
        self._frames_at_report = self.frames_processed
        self._detections_at_report = self.detections_total

    def _route_table(self, class_name: str) -> str | None:
        if self._raw_table:
            return self._raw_table
        for entry in self._routing:
            if not entry.classes or class_name in entry.classes:
                return entry.raw_table
        return None
