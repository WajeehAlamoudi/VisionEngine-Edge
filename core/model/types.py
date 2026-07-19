from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceResult:
    class_name: str
    confidence: float
    bbox:       list[float]    # [x1, y1, x2, y2] absolute pixel coords
    track_id:   str | None     # stable UUID assigned by the tracker; None when tracker not active
