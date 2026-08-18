# VisionEngine Edge — Configuration Reference

Configuration is split across 8 files, each with one responsibility.

**There are no defaults.** Every key must be present in every file. A missing
key, a wrong type, an out-of-range value, or a key the parser does not
recognise stops the device at startup with a message naming the file, the
field, and what was expected. Nothing is inferred, inherited, or silently
filled in.

Copy `config_sample/*.sample.yaml` to `config/*.yaml` (this is what
`scripts/install.sh` does) and fill in the real values.

---

## Files

| File | What it controls | Edit per deployment? |
|---|---|---|
| `device.yaml` | Device identity, log level, heartbeat, health file | Always |
| `api.yaml` | Branch id, API key, backend URL, ingest and buffer | Always |
| `cameras.yaml` | Sources, model binding, class filter, zones, tables | Always |
| `models.yaml` | Model registry — paths, devices, classes, tracker | When adding a model |
| `rules.yaml` | What is stored and what raises an alert | Per use case |
| `notifications.yaml` | Log channel and webhook delivery, per-rule routing | Per use case |
| `collection.yaml` | Dataset-building sessions | Optional |
| `botsort_tracker.yaml` | Tracker tuning — device-specific | When tracking is on |

`botsort_tracker.yaml` is not loaded by `load_config()`. It is read at model
load time from the path in `models.yaml` → `tracker`, and only when
`use_tracker: true`.

---

## The two conventions

### `"*"` means all. `[]` is an error.

Wherever a field restricts something, `"*"` lifts the restriction:

| File | Field | `"*"` means |
|---|---|---|
| `cameras.yaml` | `classes` | every class the model declares |
| `cameras.yaml` | `zones` | full frame, no subdivision |
| `cameras.yaml` | `routing[].classes` | all classes — the catch-all when placed last |
| `rules.yaml` | `class` | any class |
| `rules.yaml` | `cameras` / `zones` | any camera / any zone |
| `notifications.yaml` | `rules` / `severities` | every rule / every severity |
| `collection.yaml` | `filters.classes` | any class |

An empty list is rejected everywhere it would be ambiguous with "all".
`"*"` is a decision you wrote; `[]` is usually a deleted last entry.

**Two deliberate exceptions**, where a list is a collection rather than a
filter and being empty is a working state:

- `notifications.yaml` → `webhooks: []` — alerts go to the log only
- `collection.yaml` → `sessions: []` — no dataset collection

### `null` means "not set", and the key is still required

For scalars, an explicit `null` is the equivalent of `"*"`:

- `rules.yaml` → `min_confidence:` — no floor beyond the model's own
- `rules.yaml` → `notifications_table:` — webhook fires, nothing is stored
- `collection.yaml` → `schedule.after` / `before` / `start_date` / `end_date`
- `collection.yaml` → `sampling.interval_seconds` / `frames_per_minute`

The key must still be present. Absent is an oversight; `null` is a choice.

**`collection.yaml` may be entirely empty.** It is the only file allowed to be
blank, because collection is optional — an empty file means the feature is off.
The file must still exist.

---

## Ownership — what each file may define

```
models.yaml   owns: the classes a model is expected to output, its confidence
                    floor, its weight file, and which device runs it
              not:  which camera uses it, which classes a camera watches

cameras.yaml  owns: which model a camera uses, which of that model's classes
                    this camera watches, zones, destination tables, fps
              not:  class names or thresholds (those come from models.yaml)

rules.yaml    owns: what reaches the database at all, and what raises an alert
              not:  class definitions (models.yaml), zone definitions (cameras.yaml)

notifications.yaml
              owns: where an alert that already fired is delivered
              not:  which detections are stored, which rules fire
```

There is **no override hierarchy**. Values do not cascade between files. A
camera cannot override a model's confidence threshold — scope a rule to that
camera and set `min_confidence` on it instead.

---

## How a detection flows

```
cameras.yaml   source → frame
models.yaml    model + device → detections (filtered to the camera's classes)
cameras.yaml   zones → each detection tagged with a zone name
rules.yaml     matched at least one enabled rule?
                 no  → discarded, stored nowhere
                 yes → written to the camera's raw_table / routing target
                       and, if the rule has notify: true and cleared its
                       cooldown, to that rule's notifications_table
notifications.yaml
               the alert is delivered to the log and to every webhook whose
               rules and severities filters accept it
```

**`rules.yaml` is the gate for storage, not just for alerts.** A class no
enabled rule mentions is detected and then thrown away. If vehicles are absent
from your `detections` table, it is because no rule matches them — add a rule
with `notify: false` to store them silently.

At least one rule must be enabled. With none, the filter inverts and *every*
detection is stored, so an all-disabled file is rejected.

---

## `detections` vs `notifications`

Both tables are written from the same rule evaluation, under different
conditions:

| | `detections` | `notifications` |
|---|---|---|
| Written when | any enabled rule matched | that rule also has `notify: true`, cleared its cooldown, and sets `notifications_table` |
| Volume | every frame, every object | deduplicated by cooldown |
| Purpose | the analytical record | the actionable event log |

The **notifications table** is populated through the durable buffered ingest
path. The **webhook** is a separate, live-only path: it broadcasts to the
frontend over WebSocket, is never persisted, and is never retried. A webhook
outage loses live alerts, not stored rows.

Cooldown is keyed on `(rule, camera, zone)` — **not** on `track_id`. Two people
entering a zone within the cooldown window produce one notification. Count
people with `COUNT(DISTINCT track_id)` on `detections`, never from
`notifications`.

---

## Cross-file references

Checked at startup. A reference to something that exists nowhere is a **typo**
and fails; a reference to something real but currently disabled is a
**deliberate state** and only warns.

| Reference | Must exist in |
|---|---|
| `cameras.yaml` → `model_id` | `models.yaml` ids |
| `cameras.yaml` → `classes` | that model's `classes` |
| `rules.yaml` → `class` | some model's `classes` |
| `rules.yaml` → `cameras` | `cameras.yaml` ids |
| `rules.yaml` → `zones` | zone names on some camera |
| `notifications.yaml` → `rules` | `rules.yaml` names |
| `collection.yaml` → `camera` | `cameras.yaml` ids |
| `collection.yaml` → `filters.classes` | that camera's active classes |

Files must also exist on disk, but only for models an **enabled** camera uses —
declaring more models than the device has weights for is supported:

- `models.yaml` → `path`
- `models.yaml` → `tracker`, when `use_tracker: true`

---

## Zones

- Names use letters, digits, and underscores only: `main_entrance`, not
  `"main entrance"`. Rules reference them by exact name.
- A polygon needs at least 3 points, each `[x, y]` in pixels, non-negative.
- Zones are evaluated **in order and the first match wins**, so a frame-wide
  zone must come last or it will shadow everything after it.
- Names must be unique within a camera. The same name on two cameras is
  independent.
- For `person`, zone membership is tested at the **bottom-centre** of the
  bounding box — where the feet are. Draw zone polygons on the floor, not
  around a standing person's height. Every other class uses the box centre.
- `zones: "*"` means no subdivision; everything is tagged `full_frame`.
  Otherwise a detection outside every polygon is tagged `unzoned` — which is a
  real zone value that `rules.yaml` → `zones: "*"` still matches.

---

## Classes

- Names must match exactly what the model outputs — lowercase for COCO models
  (`person`, `car`, `truck`), or your training config's names for custom weights.
- `models.yaml` → `classes` is the source of truth. It takes no `"*"`: it is
  what camera filters are validated against, and resolving `"*"` would require
  loading the model before the config could be checked.
- Declare the classes you actually use, not the model's full list. Treat it as
  a contract — if the wrong weights file is deployed, the declared classes will
  not match.
- `cameras.yaml` → `classes` must be a subset of the model's list, or `"*"`.

---

## Adding a model

1. Put the weight file on the device, conventionally under `models/`.
2. Add an entry to `models.yaml` with a unique `id`, its `path`, a `device` the
   format can run on, and the `classes` you expect from it.
3. Reference the `id` from a camera's `model_id`.
4. Restart.

A TensorRT `.engine` must be exported on the device that will run it — it is
built for that exact GPU and TensorRT version. A Hailo `.hef` is the opposite:
it is compiled on an x86 machine and copied to the Pi.

---

## When a config is wrong

Every problem in a file is reported at once, so one restart shows the whole
list rather than one error per attempt:

```
Config error in cameras.yaml (3 problems):
  cameras[0].fps_target - expected an integer >= 1, got 0
  cameras[1].id         - expected letters, digits, hyphens, and underscores only, got "cam 02"
  cameras[1].classes    - expected "*" or a non-empty list, got an empty list
```

Cross-file problems are reported after all files parse, separated into errors
that stop startup and warnings that do not:

```
WARNING  rules[track_persons]: camera 'cam-04' is disabled, so this rule will not fire for it
ERROR    rules[truck_alerts]: camera 'cam-99' does not exist in cameras.yaml (defined: cam-01, cam-02)
```
