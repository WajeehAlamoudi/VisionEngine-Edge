from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DwellRecord:
    """Active dwell being tracked — not yet flushed to DB."""
    zone:        str
    entry_ts:    str    # ISO 8601 UTC, when object entered the zone
    last_seen:   float  # time.monotonic() — used to detect stale tracks


@dataclass
class OccupancyRecord:
    """Live occupancy state per zone+class."""
    track_ids: set   # active track_ids currently in this zone for this class
