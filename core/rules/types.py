from __future__ import annotations

from dataclasses import dataclass, field

from core.config import RuleConfig


@dataclass
class DetectionEvent:
    """
    Fully enriched detection — central contract passed between pipeline stages.

    Produced by core/pipeline/enricher.py after inference and zone assignment.
    Consumed by: rules engine, notifier, buffer.

    All downstream modules work from this object alone — no extra context needed.
    """
    # ── model output ──────────────────────────────────────────────────────────
    track_id:   str | None     # stable UUID from the tracker; None when tracker not active for this camera
    class_name: str
    confidence: float
    bbox:       list[float]    # [x1, y1, x2, y2] absolute pixel coords
    model_id:   str

    # ── spatial ───────────────────────────────────────────────────────────────
    anchor_x:      float       # anchor point in pixel coords
    anchor_y:      float       # person = feet (bottom-center), else bbox center
    anchor_x_norm: float       # anchor_x / frame_w  → 0.0–1.0
    anchor_y_norm: float       # anchor_y / frame_h  → 0.0–1.0

    # ── camera context ────────────────────────────────────────────────────────
    camera_id:   str
    camera_name: str
    frame_w:     int
    frame_h:     int

    # ── zone ──────────────────────────────────────────────────────────────────
    zone: str                  # zone name | "full_frame" | "unzoned"

    # ── temporal ──────────────────────────────────────────────────────────────
    capture_ts: str            # ISO 8601 UTC — grabbed at cap.read(), not post-inference

    # ── attribute model output (optional second-stage model) ──────────────────
    attributes: dict = field(default_factory=dict)   # e.g. {"gender": "female", "has_hardhat": True}


@dataclass
class RuleMatch:
    rule:      RuleConfig
    detection: DetectionEvent
    message:   str             # rule.message with placeholders resolved
