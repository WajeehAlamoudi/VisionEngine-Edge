# VisionEngine Edge — Configuration Reference

Configuration is split across 8 files, each with one responsibility.
When deploying to a new device, you only need to edit `api.yaml` and `cameras.yaml`.
Everything else has sensible defaults.

---

## Files and what they control

| File | What you edit here | Edit per deployment? |
|---|---|---|
| `device.yaml` | Device identity, FPS, heartbeat, buffer | Sometimes |
| `api.yaml` | API key, backend URL, ingest batch size | Always |
| `cameras.yaml` | Camera sources, model binding, class filter, zones | Always |
| `models.yaml` | Model registry — paths, classes, hardware device | When adding/changing models |
| `rules.yaml` | Alert rules, cooldowns, severity | Per use case |
| `notifications.yaml` | Webhook URLs, Slack, Teams | Per use case |
| `collection.yaml` | Frame sampling sessions for dataset building | Optional |

---

## Ownership — what each file is allowed to define

This is the most important rule. If you're unsure where to put something, check here first.

```
models.yaml   owns: what each model IS CAPABLE of detecting
                    (full class list, confidence floor, hardware device)
              does NOT own: which camera uses it, which classes to watch per camera

cameras.yaml  owns: which model each camera uses (model_id)
                    which classes THIS camera watches (subset filter)
                    per-camera confidence threshold
              does NOT own: model internals, class names (those come from models.yaml)

rules.yaml    owns: which class + zone combination triggers an alert
              does NOT own: class definitions (from models.yaml via cameras.yaml)
                            zone definitions (from cameras.yaml)
```

**Startup validation enforces this.** If a camera references a model that doesn't exist,
or lists a class the model doesn't know, the device refuses to start with a clear error:

```
ERROR  cam-02: model_id 'vehicle_v2' not found in models.yaml
ERROR  cam-01: class 'hardhat' not in model 'general_coco'
               (known: person, bicycle, car, motorcycle, bus, truck)
```

---

## How the files relate to each other

```
device.yaml            ← global defaults (fps, buffer, heartbeat)
    ↓ fps default into
cameras.yaml           ← per-camera overrides of fps, table names, zones
    ↓ model_id reference into
models.yaml            ← model registry (path, classes, hardware device)
    ↓ class names validated against
cameras.yaml classes   ← per-camera class filter (subset of model's classes)
    ↓ zone names referenced in
rules.yaml             ← class + zone → alert
    ↓ when a rule fires
notifications.yaml     ← sends to configured channels
    ↓ alert row written to
alerts_table           ← defined per-rule in rules.yaml
```

---

## Override hierarchy

Settings cascade — more specific always wins:

```
models.yaml  confidence_threshold: 0.5    ← model-level floor
cameras.yaml confidence_threshold: 0.75   ← overrides for this camera only
rules.yaml   min_confidence: 0.88         ← overrides for this rule only (must be >= model floor)
```

Same for fps:

```
device.yaml  fps_target: 5                ← all cameras
cameras.yaml fps_target: 3                ← this camera only
```

---

## Minimum config to get running

```yaml
# api.yaml
api:
  key: "cvp-your_branch_api_key"
  url: "https://your-app.ondigitalocean.app/api/v1"

# cameras.yaml — at minimum one camera
cameras:
  - id: cam-01
    name: "Main Camera"
    source: 0
    enabled: true
    model_id: general_coco      # must match an id in models.yaml
    raw_table: "detections"
    zones: []
```

Everything else uses defaults and the device will start ingesting data.

---

## Adding a new model

1. Add an entry to `models.yaml` with a unique `id`, the `path` to the weights file, and its `classes` list
2. Reference the new `id` via `model_id` on any camera in `cameras.yaml`
3. Restart the device — the new model is loaded at startup

No other files need to change.

---

## Universal convention — empty list always means ALL

This convention is consistent across every config file. When you see `[]`, it always means "no restriction — include everything":

| File | Field | `[]` means |
|---|---|---|
| `cameras.yaml` | `zones: []` | full frame — all zones |
| `cameras.yaml` | `classes: []` | all classes the model detects |
| `cameras.yaml` | `routing classes: []` | all classes — acts as fallback when placed last |
| `rules.yaml` | `zones: []` | any zone on any camera |
| `collection.yaml` | `filters.classes: []` | save frames regardless of class |

Never interpret `[]` as "none" or "disabled" — use `enabled: false` to disable something.

---

## Key rules about zone names

- Use underscores: `main_entrance` not `"main entrance"`
- Zone names in `rules.yaml` must exactly match zone names in `cameras.yaml`
- The same zone name on two different cameras is independent — each fires rules separately
- Empty zones list `[]` means full frame — everything is tagged `full_frame`

---

## Key rules about class names

- Class names in `cameras.yaml` must exactly match class names in `models.yaml`
- Class names in `rules.yaml` must match class names actually used in `cameras.yaml`
- For standard YOLOv8 / YOLO11 COCO models: use lowercase — `person`, `car`, `truck`
- For custom fine-tuned models: use whatever names your training config defines
- Startup validation catches mismatches before any camera starts
