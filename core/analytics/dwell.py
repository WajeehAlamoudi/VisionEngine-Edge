from __future__ import annotations

import time

from core.rules import DetectionEvent
from .rows import dwell_row
from .types import DwellRecord


class DwellTracker:
    """
    Tracks how long each object stays in each zone.

    State: per (track_id, class_name) → DwellRecord (zone + entry time).

    Events produced:
      - Zone change:  object moved to a different zone → close old dwell, start new one
      - Stale flush:  object not seen for dwell_timeout_seconds → close dwell

    A dwell row is only emitted when a dwell ends, not on every frame.
    One row = one complete visit: {camera_id, track_id, class, zone, entry_ts, exit_ts}
    """

    def __init__(self, camera_id: str, table: str, timeout_seconds: int) -> None:
        self._camera_id = camera_id
        self._table = table
        self._timeout = timeout_seconds
        # (track_id, class_name) → DwellRecord
        self._active: dict[tuple[int, str], DwellRecord] = {}

    def update(self, event: DetectionEvent) -> list[dict]:
        """Process one detection event. Returns completed dwell rows (may be empty)."""
        if event.track_id is None:
            return []

        key = (event.track_id, event.class_name)
        rows = []

        if key in self._active:
            rec = self._active[key]
            if rec.zone != event.zone:
                # object moved to a different zone → close old dwell
                rows.append(self._make_row(key, rec, event.capture_ts))
                # start fresh in new zone
                self._active[key] = DwellRecord(
                    zone=event.zone,
                    entry_ts=event.capture_ts,
                    last_seen=time.monotonic(),
                )
            else:
                rec.last_seen = time.monotonic()
        else:
            self._active[key] = DwellRecord(
                zone=event.zone,
                entry_ts=event.capture_ts,
                last_seen=time.monotonic(),
            )

        return rows

    def flush_stale(self, current_ts: str) -> list[dict]:
        """Close dwells for objects not seen recently. Called on periodic flush."""
        now = time.monotonic()
        rows = []
        stale = [k for k, rec in self._active.items() if now - rec.last_seen > self._timeout]
        for key in stale:
            rows.append(self._make_row(key, self._active.pop(key), current_ts))
        return rows

    def _make_row(self, key: tuple[int, str], rec: DwellRecord, exit_ts: str) -> dict:
        track_id, class_name = key
        return {
            "table": self._table,
            "row":   dwell_row(self._camera_id, track_id, class_name, rec.zone, rec.entry_ts, exit_ts),
        }
