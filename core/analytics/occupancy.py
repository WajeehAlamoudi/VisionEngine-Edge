from __future__ import annotations

import time
from collections import defaultdict

from core.rules import DetectionEvent
from .rows import occupancy_row


class OccupancyCounter:
    """
    Maintains a live count of objects per zone per class.

    Update: called on every rule-matched detection event — records which
            track_ids are currently present in each zone.
    Flush:  called periodically — emits one snapshot row per active zone+class,
            after removing track_ids that haven't been seen recently (stale).

    One row = one snapshot: {camera_id, zone, class, count, ts}
    """

    _STALE_TIMEOUT = 5.0   # seconds — object removed from zone count if not seen

    def __init__(self, camera_id: str, table: str) -> None:
        self._camera_id = camera_id
        self._table = table
        # zone → class_name → set[track_id]
        self._present: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        # track_id → last seen monotonic
        self._last_seen: dict[int, float] = {}

    def update(self, event: DetectionEvent) -> None:
        if event.track_id is None:
            return
        self._present[event.zone][event.class_name].add(event.track_id)
        self._last_seen[event.track_id] = time.monotonic()

    def flush(self, ts: str) -> list[dict]:
        """Remove stale tracks then emit one snapshot row per active zone+class."""
        self._purge_stale()

        rows = []
        for zone, class_map in self._present.items():
            for class_name, track_ids in class_map.items():
                if track_ids:
                    rows.append({
                        "table": self._table,
                        "row":   occupancy_row(self._camera_id, zone, class_name, len(track_ids), ts),
                    })
        return rows

    def _purge_stale(self) -> None:
        now = time.monotonic()
        stale = {tid for tid, t in self._last_seen.items() if now - t > self._STALE_TIMEOUT}
        if not stale:
            return
        for class_map in self._present.values():
            for track_ids in class_map.values():
                track_ids -= stale
        for tid in stale:
            del self._last_seen[tid]
