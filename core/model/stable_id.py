from __future__ import annotations

import uuid


class StableIdMap:
    """
    Maps a tracker's raw integer id to a UUID that is safe to persist.

    Every tracker in use here numbers its tracks with a counter that restarts
    at the beginning of each process — boxmot's starts at 1, DeepStream's
    NvTracker at 0. Writing that integer to the detections table as-is would
    let a track recorded today collide with an unrelated track recorded after
    the next restart, and nothing downstream could tell them apart.

    Assigning a fresh UUID the first time each raw id is seen removes that
    risk: a new map is created with every tracker (or tracking detector)
    instance, so ids from a previous run can never resurface.

    One instance per camera. Not thread-safe, and does not need to be — a
    single camera's frames are processed one at a time.
    """

    def __init__(self) -> None:
        self._ids: dict[int, str] = {}

    def get(self, raw_id: int) -> str:
        stable = self._ids.get(raw_id)
        if stable is None:
            stable = str(uuid.uuid4())
            self._ids[raw_id] = stable
        return stable

    def __len__(self) -> int:
        return len(self._ids)
