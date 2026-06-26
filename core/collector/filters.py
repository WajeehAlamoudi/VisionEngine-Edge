from __future__ import annotations

from datetime import date, datetime

from core.config import FiltersConfig, ScheduleConfig
from core.model import InferenceResult


def _in_schedule(sched: ScheduleConfig) -> bool:
    now_dt = datetime.now()
    now_time = now_dt.time()
    now_date = now_dt.date()

    if sched.start_date:
        if now_date < date.fromisoformat(sched.start_date):
            return False

    if sched.end_date:
        if now_date > date.fromisoformat(sched.end_date):
            return False

    if sched.after:
        after = datetime.strptime(sched.after, "%H:%M").time()
        if now_time < after:
            return False

    if sched.before:
        before = datetime.strptime(sched.before, "%H:%M").time()
        if now_time > before:
            return False

    return True


def _apply_filters(
        results: list[InferenceResult],
        filters: FiltersConfig,
) -> list[InferenceResult]:
    out = []
    for r in results:
        if filters.classes and r.class_name not in filters.classes:
            continue
        if filters.min_confidence > 0 and r.confidence < filters.min_confidence:
            continue
        out.append(r)
    return out
