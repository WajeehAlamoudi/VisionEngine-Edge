from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import date, datetime, timezone
from pathlib import Path

import cv2

from core.config import AppConfig, CollectionSession, FiltersConfig, ScheduleConfig
from core.model import InferenceResult

log = logging.getLogger(__name__)


class Collector:
    """
    Dataset-building observer. Receives frames from the pipeline and saves
    qualifying ones to disk based on each session's schedule, sampling mode,
    and filters.

    Does NOT open camera streams — frames are shared from the pipeline,
    so there is no additional camera load.

    Called from the pipeline after each inference cycle:
        await collector.on_frame(camera_id, frame, results, ts)
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._output_dir = Path(cfg.collection.output_dir)
        self._loop = None

        # group sessions by camera_id for O(1) lookup per frame
        self._by_camera: dict[str, list[CollectionSession]] = {}
        for session in cfg.enabled_sessions:
            self._by_camera.setdefault(session.camera, []).append(session)

        # per-session runtime state
        self._frames_saved: dict[str, int] = {}  # session_id → saved count
        self._last_saved: dict[str, float] = {}  # session_id → monotonic timestamp

    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        if self._by_camera:
            log.info(
                "collector: active for cameras: %s",
                ", ".join(sorted(self._by_camera)),
            )

    async def on_frame(
            self,
            camera_id: str,
            frame,  # numpy array (BGR, from OpenCV)
            results: list[InferenceResult],
            ts: str,  # ISO 8601 UTC timestamp
    ) -> None:
        """Called by the pipeline after every inference cycle."""
        sessions = self._by_camera.get(camera_id)
        if not sessions:
            return
        for session in sessions:
            await self._handle(session, frame, results, ts)

    # ── per-session logic ─────────────────────────────────────────────────────

    async def _handle(
            self,
            session: CollectionSession,
            frame,
            results: list[InferenceResult],
            ts: str,
    ) -> None:
        # max_frames limit (0 = unlimited)
        saved = self._frames_saved.get(session.id, 0)
        if session.max_frames > 0 and saved >= session.max_frames:
            return

        if not _in_schedule(session.schedule):
            return

        qualifying = _apply_filters(results, session.filters)

        # min_detections filter
        if session.filters.min_detections > 0 and len(qualifying) < session.filters.min_detections:
            return

        if not self._should_sample(session, qualifying):
            return

        # save in thread pool — cv2.imwrite is synchronous
        session_dir = self._output_dir / session.id
        stem = _filename_stem(ts, session.camera)

        await self._loop.run_in_executor(
            None, _save_files, session_dir, stem, frame, qualifying, session.save
        )

        self._frames_saved[session.id] = saved + 1
        self._last_saved[session.id] = time.monotonic()

        log.debug(
            "collector [%s]: saved frame %d/%s  detections=%d",
            session.id,
            self._frames_saved[session.id],
            str(session.max_frames) if session.max_frames else "∞",
            len(qualifying),
        )

    def _should_sample(self, session: CollectionSession, qualifying: list) -> bool:
        mode = session.sampling.mode
        last = self._last_saved.get(session.id, 0.0)
        now = time.monotonic()

        if mode == "interval":
            interval = session.sampling.interval_seconds or 10
            return (now - last) >= interval

        if mode == "on_detection":
            return len(qualifying) > 0

        if mode == "random":
            fpm = session.sampling.frames_per_minute or 1
            avg_interval = 60.0 / fpm
            elapsed = now - last
            if elapsed < avg_interval * 0.5:
                return False
            # add randomness: save when elapsed passes a random threshold
            # centered around avg_interval, spread ±50%
            threshold = avg_interval * random.uniform(0.5, 1.5)
            return elapsed >= threshold

        return False


# ── schedule check ────────────────────────────────────────────────────────────

def _in_schedule(sched: ScheduleConfig) -> bool:
    now_dt = datetime.now()
    now_time = now_dt.time()
    now_date = now_dt.date()

    if sched.start_date:
        if now_date < date.fromisoformat(sched.start_date):
            return False

    if sched.end_date:
        if now_date > date.fromisoformat(sched.end_date):
            return False

    if sched.after:
        after = datetime.strptime(sched.after, "%H:%M").time()
        if now_time < after:
            return False

    if sched.before:
        before = datetime.strptime(sched.before, "%H:%M").time()
        if now_time > before:
            return False

    return True


# ── filter ────────────────────────────────────────────────────────────────────

def _apply_filters(
        results: list[InferenceResult],
        filters: FiltersConfig,
) -> list[InferenceResult]:
    out = []
    for r in results:
        if filters.classes and r.class_name not in filters.classes:
            continue
        if filters.min_confidence > 0 and r.confidence < filters.min_confidence:
            continue
        out.append(r)
    return out


# ── file writing (runs in thread pool) ───────────────────────────────────────

def _save_files(
        session_dir: Path,
        stem: str,
        frame,
        results: list[InferenceResult],
        save_cfg,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)

    if save_cfg.raw:
        cv2.imwrite(str(session_dir / f"{stem}_raw.jpg"), frame)

    if save_cfg.annotated:
        annotated = _draw_boxes(frame, results)
        cv2.imwrite(str(session_dir / f"{stem}.jpg"), annotated)

    if save_cfg.metadata:
        meta = {
            "ts": stem.split("_cam")[0].replace("_", "T", 1).replace("_", ":").replace("T", " "),
            "detections": [
                {
                    "class": r.class_name,
                    "confidence": round(r.confidence, 4),
                    "bbox": [round(v, 1) for v in r.bbox],
                }
                for r in results
            ],
        }
        (session_dir / f"{stem}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )


def _draw_boxes(frame, results: list[InferenceResult]):
    out = frame.copy()
    for r in results:
        x1, y1, x2, y2 = (int(v) for v in r.bbox)
        label = f"{r.class_name} {r.confidence:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _filename_stem(ts: str, camera_id: str) -> str:
    # ts is "2026-06-22T08:14:33Z" → "2026-06-22_08-14-33"
    clean = ts.replace("T", "_").replace(":", "-").rstrip("Z")
    cam = camera_id.replace("-", "").replace("_", "")
    return f"{clean}_{cam}"
