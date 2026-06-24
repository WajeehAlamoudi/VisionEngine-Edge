from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from core.config import RuleConfig

log = logging.getLogger(__name__)


# ── shared detection type ─────────────────────────────────────────────────────
# Defined here because RulesEngine is the primary consumer.
# pipeline.py, notifier.py, and collector.py all import Detection from here.

@dataclass
class Detection:
    camera_id: str
    camera_name: str
    class_name: str
    confidence: float
    bbox: list[float]   # [x1, y1, x2, y2] absolute pixel coords
    zone: str           # zone name, "full_frame", or "unzoned"
    ts: str             # ISO 8601 UTC timestamp


@dataclass
class RuleMatch:
    rule: RuleConfig
    detection: Detection
    message: str        # rule.message with all placeholders resolved


# ── rules engine ──────────────────────────────────────────────────────────────

class RulesEngine:
    """
    Evaluates detections against the enabled rules list.

    Cooldown is tracked per rule × camera × zone — each rule has its own
    independent timer. Two rules watching the same class+zone can have
    different cooldowns and fire independently.

    Uses time.monotonic() for cooldown so system clock changes don't
    cause cooldowns to expire early or extend unexpectedly.
    """

    def __init__(self, rules: list[RuleConfig]) -> None:
        self._rules = rules
        # (rule_name, camera_id, zone) → monotonic timestamp of last fire
        self._last_fired: dict[tuple[str, str, str], float] = {}

    def evaluate(self, det: Detection) -> list[RuleMatch]:
        """
        Evaluate one detection against all enabled rules.
        Returns a RuleMatch for every rule that fires.
        An empty list means no rule matched — detection is still ingested,
        just no alert or notification is triggered.
        """
        matches: list[RuleMatch] = []

        for rule in self._rules:

            # ── class filter ─────────────────────────────────────────────────
            if rule.class_name != det.class_name:
                continue

            # ── zone filter: [] = all zones ───────────────────────────────
            if rule.zones and det.zone not in rule.zones:
                continue

            # ── confidence filter ─────────────────────────────────────────
            if rule.min_confidence is not None and det.confidence < rule.min_confidence:
                continue

            # ── cooldown check ────────────────────────────────────────────
            # key is per-rule so two rules on the same class+zone are independent
            key = (rule.name, det.camera_id, det.zone)
            now = time.monotonic()
            if rule.cooldown_seconds > 0:
                last = self._last_fired.get(key, 0.0)
                if (now - last) < rule.cooldown_seconds:
                    continue

            # ── rule fires ────────────────────────────────────────────────
            self._last_fired[key] = now
            message = _format(rule.message, det)
            matches.append(RuleMatch(rule=rule, detection=det, message=message))
            log.debug(
                "rule '%s' fired — %s in %s on %s (conf=%.2f)",
                rule.name, det.class_name, det.zone, det.camera_id, det.confidence,
            )

        return matches


def _format(template: str, det: Detection) -> str:
    return (
        template
        .replace("{class}",      det.class_name)
        .replace("{zone}",       det.zone)
        .replace("{camera}",     det.camera_id)
        .replace("{confidence}", f"{det.confidence:.2f}")
    )
