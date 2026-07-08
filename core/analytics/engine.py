from __future__ import annotations

from core.config import CameraConfig
from core.rules import DetectionEvent
from .dwell import DwellTracker
from .occupancy import OccupancyCounter
from .trajectory import TrajectorySampler


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

