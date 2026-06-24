from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ScheduleConfig:
    after: str | None           # HH:MM — time of day window start
    before: str | None          # HH:MM — time of day window end
    start_date: str | None      # YYYY-MM-DD — calendar range start
    end_date: str | None        # YYYY-MM-DD — calendar range end


@dataclass
class SamplingConfig:
    mode: str                   # interval | on_detection | random
    interval_seconds: int | None
    frames_per_minute: int | None


@dataclass
class FiltersConfig:
    classes: list[str]          # [] = all classes (universal convention)
    min_confidence: float
    min_detections: int


@dataclass
class SaveConfig:
    annotated: bool             # frame with bounding boxes drawn
    raw: bool                   # clean frame, no annotations
    metadata: bool              # JSON sidecar with detection details


@dataclass
class CollectionSession:
    id: str
    camera: str
    enabled: bool
    schedule: ScheduleConfig
    sampling: SamplingConfig
    filters: FiltersConfig
    save: SaveConfig
    max_frames: int             # 0 = unlimited


@dataclass
class CollectionConfig:
    output_dir: str
    sessions: list[CollectionSession]


def parse(raw_list: list) -> CollectionConfig:
    # collection.yaml is a list where one item holds output_dir
    # and the rest are session definitions
    output_dir = "./collected"
    sessions: list[CollectionSession] = []

    for item in raw_list:
        if "output_dir" in item and "id" not in item:
            output_dir = item["output_dir"]
            continue

        sched = item.get("schedule", {})
        samp  = item.get("sampling", {})
        filt  = item.get("filters", {})
        save  = item.get("save", {})

        sessions.append(CollectionSession(
            id=item["id"],
            camera=item["camera"],
            enabled=item.get("enabled", True),
            schedule=ScheduleConfig(
                after=sched.get("after"),
                before=sched.get("before"),
                start_date=sched.get("start_date"),
                end_date=sched.get("end_date"),
            ),
            sampling=SamplingConfig(
                mode=samp.get("mode", "interval"),
                interval_seconds=samp.get("interval_seconds"),
                frames_per_minute=samp.get("frames_per_minute"),
            ),
            filters=FiltersConfig(
                classes=filt.get("classes", []),
                min_confidence=float(filt.get("min_confidence", 0.0)),
                min_detections=int(filt.get("min_detections", 0)),
            ),
            save=SaveConfig(
                annotated=save.get("annotated", True),
                raw=save.get("raw", True),
                metadata=save.get("metadata", True),
            ),
            max_frames=int(item.get("max_frames", 0)),
        ))

    return CollectionConfig(output_dir=output_dir, sessions=sessions)
