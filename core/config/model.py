from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ModelConfig:
    id: str
    name: str
    version: str
    path: str
    device: str             # auto | cpu | cuda | hailo | coreml | mps
    classes: list[str]      # full class list — what the model CAN detect
    confidence_threshold: float
    iou_threshold: float
    input_size: list[int]   # [width, height]
    use_tracker: bool        # true = BoT-SORT tracker ON → track_id populated per object
    tracker: str             # tracker algorithm file — "botsort.yaml" | "bytetrack.yaml"


def parse(raw: dict) -> ModelConfig:
    return ModelConfig(
        id=raw["id"],
        name=raw.get("name", raw["id"]),
        version=raw.get("version", "1.0.0"),
        path=raw["path"],
        device=raw.get("device", "auto"),
        classes=raw.get("classes", []),
        confidence_threshold=float(raw.get("confidence_threshold", 0.5)),
        iou_threshold=float(raw.get("iou_threshold", 0.45)),
        input_size=raw.get("input_size", [640, 640]),
        use_tracker=bool(raw.get("use_tracker", False)),
        tracker=raw.get("tracker", "botsort.yaml"),
    )
