from __future__ import annotations

import logging
import time

from core.config import RuleConfig
from .types import DetectionEvent, RuleMatch

log = logging.getLogger(__name__)


class RulesEngine:
    """
    Two roles:

    1. Filter  — discards detections that match no rule (irrelevant to this deployment).
    2. Alert   — rules with alert=True that cleared cooldown produce RuleMatch objects.

    filter_and_tag(event) returns:
      None         → no rule matched → discard
      []           → matched, no alerts (alert=False or cooldown active)
      [m1, m2, …] → matched + these alerts should fire
    """

    def __init__(self, rules: list[RuleConfig]) -> None:
        self._rules = rules
        # (rule_name, camera_id, zone) → monotonic timestamp of last alert
        self._last_fired: dict[tuple[str, str, str], float] = {}

    def filter_and_tag(self, event: DetectionEvent) -> list[RuleMatch] | None:
        if not self._rules:
            return []   # no rules configured → all detections pass

        passes = False
        matches: list[RuleMatch] = []

        for rule in self._rules:
            if rule.cameras and event.camera_id not in rule.cameras:
                continue
            if rule.class_name != event.class_name:
                continue
            if rule.zones and event.zone not in rule.zones:
                continue
            if rule.min_confidence is not None and event.confidence < rule.min_confidence:
                continue

            passes = True

            if not rule.alert:
                continue    # filter-only rule — passes silently, no alert

            key = (rule.name, event.camera_id, event.zone)
            now = time.monotonic()
            if rule.cooldown_seconds > 0:
                last = self._last_fired.get(key, 0.0)
                if (now - last) < rule.cooldown_seconds:
                    continue

            self._last_fired[key] = now
            matches.append(RuleMatch(
                rule=rule,
                detection=event,
                message=_format(rule.message, event),
            ))
            log.debug(
                "rule '%s' fired — %s in %s on %s (conf=%.2f)",
                rule.name, event.class_name, event.zone, event.camera_id, event.confidence,
            )

        if not passes:
            return None

        return matches


def _format(template: str, det: DetectionEvent) -> str:
    return (
        template
        .replace("{class}",      det.class_name)
        .replace("{zone}",       det.zone)
        .replace("{camera}",     det.camera_id)
        .replace("{confidence}", f"{det.confidence:.2f}")
    )
