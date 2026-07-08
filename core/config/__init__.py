from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .api import ApiConfig, parse as _parse_api
from .camera import CameraConfig, parse as _parse_camera
from .collection import CollectionConfig, parse as _parse_collection
from .device import DeviceConfig, parse as _parse_device
from .model import ModelConfig, parse as _parse_model
from .notifications import NotificationsConfig, WebhookConfig, parse as _parse_notifications
from .rule import RuleConfig, parse as _parse_rule

# re-export all dataclasses so callers only need: from core.config import XxxConfig
__all__ = [
    "AppConfig", "load_config",
    "ApiConfig", "IngestConfig", "RequestConfig",
    "CameraConfig", "Zone", "RoutingEntry",
    "CollectionConfig", "CollectionSession",
    "DeviceConfig", "HeartbeatConfig", "HealthFileConfig", "BufferConfig",
    "ModelConfig",
    "NotificationsConfig", "WebhookConfig", "LogChannelConfig",
    "RuleConfig",
]

from .api import IngestConfig, RequestConfig
from .camera import Zone, RoutingEntry
from .collection import CollectionSession
from .device import HeartbeatConfig, HealthFileConfig, BufferConfig
from .notifications import LogChannelConfig
from .collection import ScheduleConfig, SamplingConfig, FiltersConfig, SaveConfig


# ── root config ───────────────────────────────────────────────────────────────

@dataclass
class AppConfig:
    device: DeviceConfig
    api: ApiConfig
    models: dict[str, ModelConfig]  # keyed by model id
    cameras: list[CameraConfig]
    rules: list[RuleConfig]
    notifications: NotificationsConfig
    collection: CollectionConfig

    def get_camera(self, camera_id: str) -> CameraConfig | None:
        return next((c for c in self.cameras if c.id == camera_id), None)

    def get_model(self, model_id: str) -> ModelConfig | None:
        return self.models.get(model_id)

    @property
    def enabled_cameras(self) -> list[CameraConfig]:
        return [c for c in self.cameras if c.enabled]

    @property
    def enabled_rules(self) -> list[RuleConfig]:
        return [r for r in self.rules if r.enabled]

    @property
    def enabled_webhooks(self) -> list[WebhookConfig]:
        return [w for w in self.notifications.webhooks if w.enabled]

    @property
    def enabled_sessions(self) -> list[CollectionSession]:
        return [s for s in self.collection.sessions if s.enabled]


# ── yaml helper ───────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── validation ────────────────────────────────────────────────────────────────

def _validate(cfg: AppConfig) -> None:
    errors: list[str] = []
    warnings: list[str] = []

    # max_cameras
    enabled_count = len(cfg.enabled_cameras)
    if enabled_count > cfg.device.max_cameras:
        errors.append(
            f"device.max_cameras is {cfg.device.max_cameras} but "
            f"{enabled_count} cameras are enabled"
        )

    for cam in cfg.cameras:
        p = f"cameras[{cam.id}]"

        model = cfg.models.get(cam.model_id)
        if model is None:
            errors.append(
                f"{p}: model_id '{cam.model_id}' not found in models.yaml "
                f"(available: {', '.join(cfg.models) or 'none'})"
            )
            continue

        # resolve classes: [] = all from model
        if not cam.classes:
            cam.classes = list(model.classes)

        # camera classes must be a subset of model classes
        model_class_set = set(model.classes)
        unknown = [c for c in cam.classes if c not in model_class_set]
        if unknown:
            errors.append(
                f"{p}: class(es) {unknown} not in model '{model.id}' "
                f"(known: {', '.join(model.classes)})"
            )

        # resolve confidence_threshold: None = inherit model floor
        if cam.confidence_threshold is None:
            cam.confidence_threshold = model.confidence_threshold
        elif cam.confidence_threshold < model.confidence_threshold:
            errors.append(
                f"{p}: confidence_threshold {cam.confidence_threshold} is below "
                f"model '{model.id}' floor of {model.confidence_threshold}"
            )

        # raw_table and routing are mutually exclusive
        if cam.raw_table and cam.routing:
            errors.append(f"{p}: use raw_table OR routing, not both")

        if not cam.raw_table and not cam.routing and cam.analytics.raw:
            warnings.append(
                f"{p}: no raw_table or routing defined — detections will not be stored"
            )

        if cam.routing:
            covered: set[str] = set()
            for entry in cam.routing:
                if not entry.classes:
                    covered.update(cam.classes)
                else:
                    bad = [c for c in entry.classes if c not in model_class_set]
                    if bad:
                        errors.append(
                            f"{p} routing: class(es) {bad} not in model '{model.id}'"
                        )
                    covered.update(entry.classes)

            uncovered = set(cam.classes) - covered
            if uncovered:
                warnings.append(
                    f"{p}: class(es) {sorted(uncovered)} have no routing entry — "
                    f"those detections will be skipped. "
                    f"Add a '- classes: []' catch-all entry."
                )

    # collect active classes and zone names from enabled cameras
    active_classes: set[str] = set()
    active_zones: set[str] = set()
    for cam in cfg.enabled_cameras:
        active_classes.update(cam.classes)
        for zone in cam.zones:
            active_zones.add(zone.name)

    enabled_camera_ids = {c.id for c in cfg.enabled_cameras}
    for rule in cfg.enabled_rules:
        # camera filter validation
        for cam_id in rule.cameras:
            if cam_id not in enabled_camera_ids:
                warnings.append(
                    f"rules[{rule.name}]: camera '{cam_id}' not found in enabled cameras "
                    f"(available: {', '.join(sorted(enabled_camera_ids)) or 'none'})"
                )

        if rule.class_name not in active_classes:
            warnings.append(
                f"rules[{rule.name}]: class '{rule.class_name}' not active on any "
                f"enabled camera (active: {', '.join(sorted(active_classes)) or 'none'})"
            )
        for zone in rule.zones:
            if zone and zone not in active_zones:
                warnings.append(
                    f"rules[{rule.name}]: zone '{zone}' not found on any enabled camera"
                )

    # collection session camera ids must exist
    camera_ids = {c.id for c in cfg.cameras}
    for session in cfg.collection.sessions:
        if session.camera not in camera_ids:
            errors.append(
                f"collection[{session.id}]: camera '{session.camera}' "
                f"not found in cameras.yaml"
            )

    for w in warnings:
        print(f"WARNING  {w}", file=sys.stderr)

    if errors:
        if warnings:
            print("", file=sys.stderr)
        for e in errors:
            print(f"ERROR    {e}", file=sys.stderr)
        print("", file=sys.stderr)
        raise SystemExit(
            f"Config validation failed ({len(errors)} error(s)) — "
            f"fix the errors above and restart."
        )


# ── public entry point ────────────────────────────────────────────────────────

def load_config(config_dir: Path | str = "config") -> AppConfig:
    """Load and validate all config files. Raises SystemExit on any config error."""
    d = Path(config_dir)

    device = _parse_device(_load_yaml(d / "device.yaml")["device"])
    api = _parse_api(_load_yaml(d / "api.yaml")["api"])
    models = {m["id"]: _parse_model(m) for m in _load_yaml(d / "models.yaml")["models"]}
    cameras = [_parse_camera(c, device.fps_target) for c in _load_yaml(d / "cameras.yaml")["cameras"]]
    rules = [_parse_rule(r) for r in _load_yaml(d / "rules.yaml")["rules"]]
    notifications = _parse_notifications(_load_yaml(d / "notifications.yaml")["notifications"])
    collection = _parse_collection(_load_yaml(d / "collection.yaml")["collection"])

    cfg = AppConfig(
        device=device,
        api=api,
        models=models,
        cameras=cameras,
        rules=rules,
        notifications=notifications,
        collection=collection,
    )

    _validate(cfg)
    return cfg
