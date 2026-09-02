from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from .api import ApiConfig, parse as _parse_api
from .camera import CameraConfig, parse_all as _parse_cameras
from .collection import CollectionConfig, parse as _parse_collection
from .device import DeviceConfig, parse as _parse_device
from .model import ModelConfig, parse_all as _parse_models
from .notifications import NotificationsConfig, WebhookConfig, parse as _parse_notifications
from .rule import RuleConfig, parse_all as _parse_rules
from .strict import ALL, ConfigError, load_section

# re-export all dataclasses so callers only need: from core.config import XxxConfig
__all__ = [
    "AppConfig", "load_config", "ConfigError",
    "ApiConfig", "IngestConfig", "BufferConfig", "RequestConfig",
    "CameraConfig", "Zone", "RoutingEntry",
    "CollectionConfig", "CollectionSession",
    "DeviceConfig", "HeartbeatConfig", "HealthFileConfig",
    "ModelConfig",
    "NotificationsConfig", "WebhookConfig", "LogChannelConfig",
    "RuleConfig",
]

from .api import IngestConfig, BufferConfig, RequestConfig
from .camera import Zone, RoutingEntry
from .collection import CollectionSession
from .device import HeartbeatConfig, HealthFileConfig
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

        # resolve classes: "*" was stored as [] at parse time = all from model
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

        # raw_table/routing exclusivity and presence are enforced at parse time
        # by camera.py, which can distinguish an absent key from an empty value.

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
                    f"{p}: class(es) {sorted(uncovered)} have no routing entry - "
                    f"those detections will be skipped. "
                    f"Add a '- classes: \"*\"' catch-all entry."
                )

    # Weight and tracker files must exist for any model an enabled camera uses.
    # Models that nothing enables are never loaded by ModelRegistry, so a
    # missing file there is legitimate — declaring more models than the device
    # has weights for is a supported pattern.
    for model_id in sorted({c.model_id for c in cfg.enabled_cameras}):
        model = cfg.models.get(model_id)
        if model is None:
            continue    # already reported by the camera loop above

        # .mlpackage is a directory, so exists() rather than is_file()
        if not Path(model.path).exists():
            errors.append(
                f"models[{model.id}]: path '{model.path}' does not exist "
                f"(an enabled camera uses this model)"
            )

        # Required only when tracking is on, for every device including
        # deepstream - which builds no nvtracker at all when use_tracker is
        # false, and so never reads this file.
        if model.use_tracker and not Path(model.tracker).is_file():
            if model.device == "deepstream":
                errors.append(
                    f"models[{model.id}]: tracker file '{model.tracker}' does not "
                    f"exist - nvtracker reads it to decide which algorithm to run, "
                    f"and has no default. Copy one of "
                    f"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/"
                    f"config_tracker_*.yml"
                )
            else:
                errors.append(
                    f"models[{model.id}]: use_tracker is true but the tracker file "
                    f"'{model.tracker}' does not exist - tracking would silently "
                    f"fall back to defaults with ReID disabled"
                )

        if model.device == "deepstream" and model.ds_infer_config:
            if not Path(model.ds_infer_config).is_file():
                errors.append(
                    f"models[{model.id}]: ds_infer_config '{model.ds_infer_config}' "
                    f"does not exist - nvinfer cannot start without it "
                    f"(an enabled camera uses this model)"
                )

    # collect active classes and zone names from enabled cameras
    active_classes: set[str] = set()
    active_zones: set[str] = set()
    for cam in cfg.enabled_cameras:
        active_classes.update(cam.classes)
        for zone in cam.zones:
            active_zones.add(zone.name)

    # A rule referencing something that exists nowhere is a typo, and the rule
    # would silently never fire — that is an error. A rule referencing
    # something real but currently disabled is a deliberate state, so it only
    # warns.
    all_camera_ids = {c.id for c in cfg.cameras}
    enabled_camera_ids = {c.id for c in cfg.enabled_cameras}
    all_zone_names = {z.name for c in cfg.cameras for z in c.zones}
    all_model_classes = {cls for m in cfg.models.values() for cls in m.classes}

    for rule in cfg.enabled_rules:
        for cam_id in rule.cameras:
            if cam_id not in all_camera_ids:
                errors.append(
                    f"rules[{rule.name}]: camera '{cam_id}' does not exist in cameras.yaml "
                    f"(defined: {', '.join(sorted(all_camera_ids)) or 'none'})"
                )
            elif cam_id not in enabled_camera_ids:
                warnings.append(
                    f"rules[{rule.name}]: camera '{cam_id}' is disabled, "
                    f"so this rule will not fire for it"
                )

        if rule.class_name != ALL:
            if rule.class_name not in all_model_classes:
                errors.append(
                    f"rules[{rule.name}]: class '{rule.class_name}' is not declared by any "
                    f"model in models.yaml (declared: {', '.join(sorted(all_model_classes)) or 'none'})"
                )
            elif rule.class_name not in active_classes:
                warnings.append(
                    f"rules[{rule.name}]: class '{rule.class_name}' is not active on any "
                    f"enabled camera (active: {', '.join(sorted(active_classes)) or 'none'})"
                )

        for zone in rule.zones:
            if zone not in all_zone_names:
                errors.append(
                    f"rules[{rule.name}]: zone '{zone}' is not defined on any camera "
                    f"(defined: {', '.join(sorted(all_zone_names)) or 'none'})"
                )
            elif zone not in active_zones:
                warnings.append(
                    f"rules[{rule.name}]: zone '{zone}' exists only on a disabled camera"
                )

    # Webhook rule filters must name rules that exist. A typo means the webhook
    # silently never fires, which is the failure this validation exists to stop.
    all_rule_names = {r.name for r in cfg.rules}
    enabled_rule_names = {r.name for r in cfg.enabled_rules}

    for webhook in cfg.notifications.webhooks:
        if not webhook.enabled:
            continue

        for rule_name in webhook.rules:
            if rule_name not in all_rule_names:
                errors.append(
                    f"notifications[{webhook.name}]: rule '{rule_name}' does not exist "
                    f"in rules.yaml (defined: {', '.join(sorted(all_rule_names)) or 'none'})"
                )
            elif rule_name not in enabled_rule_names:
                warnings.append(
                    f"notifications[{webhook.name}]: rule '{rule_name}' is disabled, "
                    f"so this webhook will not fire for it"
                )

        if webhook.rules and not (set(webhook.rules) & enabled_rule_names):
            warnings.append(
                f"notifications[{webhook.name}]: none of its rules are enabled - "
                f"this webhook can never fire"
            )

    # collection session camera ids must exist
    for session in cfg.collection.sessions:
        if session.camera not in all_camera_ids:
            errors.append(
                f"collection[{session.id}]: camera '{session.camera}' does not exist "
                f"in cameras.yaml (defined: {', '.join(sorted(all_camera_ids)) or 'none'})"
            )
            continue
        if session.enabled and session.camera not in enabled_camera_ids:
            warnings.append(
                f"collection[{session.id}]: camera '{session.camera}' is disabled, "
                f"so this session will collect nothing"
            )

        cam = cfg.get_camera(session.camera)
        unknown = [c for c in session.filters.classes if c not in (cam.classes if cam else [])]
        if unknown:
            errors.append(
                f"collection[{session.id}]: filter class(es) {unknown} are not active on "
                f"camera '{session.camera}' (active: {', '.join(cam.classes) if cam else 'none'})"
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
            f"Config validation failed ({len(errors)} error(s)) - "
            f"fix the errors above and restart."
        )


# ── public entry point ────────────────────────────────────────────────────────

def load_config(config_dir: Path | str = "config") -> AppConfig:
    """Load and validate all config files. Raises SystemExit on any config error."""
    d = Path(config_dir)

    device = _parse_device(load_section(d / "device.yaml", "device"))
    # api.yaml is on the strict loader — missing file, empty file, or a missing
    # top-level section each raise ConfigError naming the file, instead of the
    # TypeError/KeyError the untyped path above still produces.
    api = _parse_api(load_section(d / "api.yaml", "api"))
    models = _parse_models(load_section(d / "models.yaml", "models"))
    cameras = _parse_cameras(load_section(d / "cameras.yaml", "cameras"))
    rules = _parse_rules(load_section(d / "rules.yaml", "rules"))
    notifications = _parse_notifications(load_section(d / "notifications.yaml", "notifications"))

    # collection is an optional parallel feature (dataset building) — unlike
    # the files above, a blank/empty file means "not configured", not a
    # config error, so it degrades to zero sessions instead of raising.
    collection = _parse_collection(
        load_section(d / "collection.yaml", "collection", allow_empty=True))

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
