from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path

from core.config import AppConfig, CollectionSession
from core.model import InferenceResult
from .filters import _apply_filters, _in_schedule
from .writer import _filename_stem, _save_files, _write_dataset_yaml

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

        # per-session class list, taken from the camera's model class list so
        # label ids stay stable, training-ready, and recorded in data.yaml
        self._class_names: dict[str, list[str]] = {}
        self._class_ids: dict[str, dict[str, int]] = {}
        for session in cfg.enabled_sessions:
            names: list[str] = []
            cam = cfg.get_camera(session.camera)
            if cam:
                model = cfg.get_model(cam.model_id)
                if model and model.classes:
                    names = list(model.classes)
            self._class_names[session.id] = names
            self._class_ids[session.id] = {name: i for i, name in enumerate(names)}

        # per-session runtime state
        self._frames_saved: dict[str, int] = {}    # session_id → saved count
        self._last_saved:   dict[str, float] = {}  # session_id → monotonic timestamp

    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        if not self._by_camera:
            return

        # write each session's data.yaml up front so the dataset descriptor
        # exists before the first frame is saved
        for sessions in self._by_camera.values():
            for session in sessions:
                session_dir = self._output_dir / session.id
                _write_dataset_yaml(session_dir, self._class_names.get(session.id, []))

        log.info(
            "collector: active for cameras: %s",
            ", ".join(sorted(self._by_camera)),
        )

    async def on_frame(
            self,
            camera_id: str,
            frame,                       # numpy array (BGR, from OpenCV)
            results: list[InferenceResult],
            ts: str,                     # ISO 8601 UTC timestamp
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
        saved = self._frames_saved.get(session.id, 0)
        if session.max_frames > 0 and saved >= session.max_frames:
            return

        if not _in_schedule(session.schedule):
            return

        qualifying = _apply_filters(results, session.filters)

        if session.filters.min_detections > 0 and len(qualifying) < session.filters.min_detections:
            return

        if not self._should_sample(session, qualifying):
            return

        session_dir = self._output_dir / session.id
        stem = _filename_stem(ts, session.camera)

        await self._loop.run_in_executor(
            None, _save_files, session_dir, stem, frame, qualifying,
            self._class_ids.get(session.id, {}),
        )

        self._frames_saved[session.id] = saved + 1
        self._last_saved[session.id] = time.monotonic()

        log.info(
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
            threshold = avg_interval * random.uniform(0.5, 1.5)
            return elapsed >= threshold

        return False
