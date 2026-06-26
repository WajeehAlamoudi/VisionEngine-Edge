from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnalyticsConfig:
    # ── what to push ──────────────────────────────────────────────────────────
    raw:        bool    # push raw detection rows (one per detection per frame)
    dwell:      bool    # track how long each object stays in each zone
    occupancy:  bool    # periodic snapshot of how many objects are in each zone
    trajectory: bool    # sample position points over time (for heatmaps)
    crossing:   bool    # detect line crossings (requires lines: in cameras.yaml)

    # ── tracker ───────────────────────────────────────────────────────────────
    tracker: str        # ultralytics tracker config — built-in name or file path
                        # built-ins: "botsort.yaml", "bytetrack.yaml"
                        # custom:    "./config/my_tracker.yaml" or absolute path

    # ── output tables ─────────────────────────────────────────────────────────
    dwell_table:      str
    occupancy_table:  str
    trajectory_table: str

    # ── tuning ────────────────────────────────────────────────────────────────
    dwell_timeout_seconds:       int   # close a dwell after N seconds without detection
    trajectory_interval_seconds: int   # sample one trajectory point every N seconds

    @property
    def needs_tracker(self) -> bool:
        """True when any analytic requires object identity across frames."""
        return self.dwell or self.occupancy or self.trajectory or self.crossing
