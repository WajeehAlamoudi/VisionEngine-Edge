from __future__ import annotations

import time

from core.rules import DetectionEvent


class TrajectorySampler:
    """
    Samples object positions at a fixed interval for heatmap and path analysis.

    Emits one trajectory row every trajectory_interval_seconds per track_id —
    not on every frame, which would produce too much data.

    One row = one position sample:
      {camera_id, track_id, class, zone, anchor_x, anchor_y, anchor_x_norm, anchor_y_norm, ts}
    """

    def __init__(self, camera_id: str, table: str, interval_seconds: int) -> None:
        self._camera_id = camera_id
        self._table = table
        self._interval = float(interval_seconds)
        # track_id → monotonic timestamp of last sample
        self._last_sampled: dict[int, float] = {}

    def update(self, event: DetectionEvent) -> list[dict]:
        if event.track_id is None:
            return []

        now = time.monotonic()
        last = self._last_sampled.get(event.track_id, 0.0)

        if now - last < self._interval:
            return []

        self._last_sampled[event.track_id] = now
        return [{
            "table": self._table,
            "row": {
                "camera_id":     event.camera_id,
                "track_id":      event.track_id,
                "class":         event.class_name,
                "zone":          event.zone,
                "anchor_x":      event.anchor_x,
                "anchor_y":      event.anchor_y,
                "anchor_x_norm": event.anchor_x_norm,
                "anchor_y_norm": event.anchor_y_norm,
                "ts":            event.capture_ts,
            },
        }]
