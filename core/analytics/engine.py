from __future__ import annotations

import logging

from core.config import CameraConfig
from core.pipeline.rows import detection_row
from core.rules import DetectionEvent
from .dwell import DwellTracker
from .occupancy import OccupancyCounter
from .trajectory import TrajectorySampler

log = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Per-camera analytics orchestrator.

    Receives DetectionEvents (rule-filtered, enriched) and routes them to
    whichever analytics modules are enabled for this camera.

    process()       → called on every DetectionEvent → returns rows for the buffer
    flush_periodic() → called on the summary interval → flushes occupancy snapshots
                       and closes stale dwell sessions
    """

    def __init__(self, cam: CameraConfig) -> None:
        cfg = cam.analytics

        self._raw_enabled = cfg.raw
        self._raw_table   = cam.raw_table
        self._routing     = cam.routing

        self._dwell = (
            DwellTracker(cam.id, cfg.dwell_table, cfg.dwell_timeout_seconds)
            if cfg.dwell else None
        )
        self._occupancy = (
            OccupancyCounter(cam.id, cfg.occupancy_table)
            if cfg.occupancy else None
        )
        self._trajectory = (
            TrajectorySampler(cam.id, cfg.trajectory_table, cfg.trajectory_interval_seconds)
            if cfg.trajectory else None
        )

    def process(self, event: DetectionEvent) -> list[dict]:
        """Route one DetectionEvent through all enabled analytics modules."""
        rows: list[dict] = []

        if self._raw_enabled:
            table = self._route_table(event.class_name)
            if table:
                rows.append({"table": table, "row": detection_row(event)})

        if self._dwell:
            rows.extend(self._dwell.update(event))

        if self._occupancy:
            self._occupancy.update(event)   # occupancy is batched, not per-event

        if self._trajectory:
            rows.extend(self._trajectory.update(event))

        return rows

    def flush_periodic(self, ts: str) -> list[dict]:
        """Flush time-based outputs — occupancy snapshots and stale dwell sessions."""
        rows: list[dict] = []
        if self._occupancy:
            rows.extend(self._occupancy.flush(ts))
        if self._dwell:
            rows.extend(self._dwell.flush_stale(ts))
        return rows

    def _route_table(self, class_name: str) -> str | None:
        if self._raw_table:
            return self._raw_table
        for entry in self._routing:
            if not entry.classes or class_name in entry.classes:
                return entry.raw_table
        log.debug("no routing entry for class '%s' — raw detection not stored", class_name)
        return None
