<div align="center">

```
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗  
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝  
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
```

### **Configuration Reference**

<br/>

[![Files](https://img.shields.io/badge/Files-8-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](../config/config_sample/)
[![Validation](https://img.shields.io/badge/Defaults-none-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#no-defaults)

<br/>

> *Eight files, one responsibility each, no value inherited from anywhere.*

</div>

---

Per-field detail — ranges, units, what a specific number does — lives in the
comments beside each key in
[`config/config_sample/`](../config/config_sample/). This guide covers what each
**file** owns, how they reference each other, and the conventions that apply
across all of them.

Copy `config_sample/*.sample.yaml` to `config/*.yaml` (what `scripts/install.sh`
does) and fill in real values.

---

## No defaults

**Every key must be present in every file.** A missing key, a wrong type, an
out-of-range value, or a key the parser does not recognise stops the device at
startup with a message naming the file, the field, and what was expected.
Nothing is inferred, inherited, or silently filled in.

This is a deliberate trade. A default that is right for a prototype is a silent
wrong answer in production — the device runs, the numbers are subtly off, and
nothing points at the cause. An explicit value is a decision someone made and
can be reviewed in a diff.

---

## The files

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
| `deepstream_infer.txt` | nvinfer config — network shape, class count, clustering | `device: deepstream` only |
| `peoplenet_labels.txt` | Model's class names, one per line, in model order | `device: deepstream` only |
| `nvdcf_tracker.yml` | nvtracker config — chooses the tracking algorithm | `device: deepstream` + `use_tracker` |

`botsort_tracker.yaml` is the odd one out: it is **not** loaded by
`load_config()` and **not** strictly validated. It is read at model load time
from the path in `models.yaml` → `tracker`, and only when `use_tracker: true`.
A typo there is silently absorbed — see [Tracker config](#tracker-config).

---

## What each file is for

### `device.yaml` — who this device is

Identity (`id`, `name`, `location`) that lands in every `nodes` row, plus
operational settings: `log_level`, `max_cameras` as a guard against
over-subscribing the hardware, and two independent reporters.

`heartbeat` writes a `nodes` row to the backend on an interval. `health_file`
writes the same picture to local JSON — useful for an external watchdog that
should work even when the network is down. They are separately enabled because
they serve different consumers.

### `api.yaml` — where data goes and what happens when it can't

`branch_id` and `key` come from creating the branch in the dashboard; `url` is
the backend. The three sub-blocks describe the delivery path:

- **`ingest`** — `batch_size` and `flush_interval_seconds`: rows are sent in batches, or after the interval, whichever comes first.
- **`buffer`** — a local SQLite spool. When the backend is unreachable rows accumulate here and replay later with their original capture timestamps. `max_size_mb` is the cap at which the **oldest rows are dropped**, and `delete_after_hours` expires them regardless. Both are data-loss policy: set them for how long an outage you intend to survive.
- **`request`** — timeout and `max_consecutive_failures` before the link is treated as down.

### `cameras.yaml` — one entry per camera

`source` is a USB index (unquoted integer), an `rtsp://` URL, or a file path.
`enabled: false` keeps an entry present but inactive — the preferred way to park
a camera, since deleting it also deletes its zones.

`fps_target` is a **ceiling, not a promise**. The real rate is whatever the
hardware sustains; read it from the throughput log line
([TOOLS](TOOLS.md#reading-the-logs)).

`model_id` binds to `models.yaml`. `classes` narrows that model's output for
this camera. `zones` subdivides the frame. Destination tables are set either as
one `raw_table` or as `routing` — a list mapping class groups to different
tables, which is how one camera splits people and vehicles apart.

### `models.yaml` — the model registry

One entry per weight file: `path`, the `device` that runs it, the `classes` it
is expected to output, and its thresholds. Declaring a model costs nothing —
only models used by an **enabled** camera are loaded or checked for existence.

`use_tracker` turns on BoT-SORT for every camera bound to this model, and
`tracker` points at the tracker config. `half` requests FP16 and is meaningful
only on CUDA. Which device implies which backend is in
[ARCHITECTURE](ARCHITECTURE.md#detector-backends).

#### `device: deepstream`

NVIDIA's DeepStream SDK, on Jetson or a dGPU. Same hardware as `cuda`, a
different runtime on it: `nvinfer` for detection and `nvtracker` for tracking,
in one GStreamer pipeline. Choose option 5 in `scripts/install.sh` to set up
the dependencies — it verifies the SDK, installs the GStreamer Python
bindings, and copies a tracker config out of the installed SDK so it matches
your DeepStream version.

Three fields behave differently on this device:

| Field | On `deepstream` |
|---|---|
| `ds_infer_config` | **Required.** nvinfer's config — input shape, class count, precision, clustering. Omit the key entirely on every other device. |
| `tracker` | Chooses the **algorithm**, not just its parameters — IOU, NvSORT, NvDCF or NvDeepSORT. Required when `use_tracker` is true. |
| `use_tracker` | Means the same as everywhere else. `true` builds an `nvtracker` element into the pipeline; `false` builds none, so NvDCF never runs and `track_id` is null. |

`path` still names the engine and overrides whatever `model-engine-file` the
nvinfer config contains, so `models.yaml` stays the single source of truth for
which weights are running.

**Class names.** The nvinfer config's `labelfile-path` points at a plain text
file listing the model's classes, one per line, **in the model's own order**.
It is required, and `models.yaml` `classes` does not replace it — the two mean
different things:

| | Meaning | Order matters? | Subset allowed? |
|---|---|---|---|
| label file | what the model outputs | yes — `class_id` indexes into it | no, all of them |
| `classes` | what you want to keep | no | yes, any subset |

A ten-class model narrowed to `[person, cup]` still detects `cup` as `class_id
5`; the label file is what knows that. `install.sh` writes
`config/peoplenet_labels.txt` with PeopleNet's three names. A name in `classes`
that appears nowhere in the label file is a startup error, since it could never
match anything.

Any NVIDIA TAO model on the DetectNet_v2 architecture — PeopleNet,
TrafficCamNet, DashCamNet, FaceDetect — is a config change with no code change:
update `path`, `ds_infer_config`, `classes` and `input_size`. Models with a
different output head (PeopleNet Transformer, PeopleSegNet) need a custom
parser library named in the nvinfer config.

Requires `pyds`, which does **not** ship with DeepStream and is not on PyPI —
the installer tells you where to get a version-matched wheel.

### `rules.yaml` — the storage gate

The most consequential file, and the most commonly misread.

```
matched at least one enabled rule?
  no  → discarded, stored nowhere
  yes → written to the camera's raw_table / routing target
        and, if notify: true and the cooldown has cleared,
        to that rule's notifications_table
```

**A class no enabled rule mentions is detected and then thrown away.** If
vehicles are missing from your `detections` table, no rule matches them — add
one with `notify: false` to store them silently.

At least one rule must be enabled. With none, the filter **inverts** and *every*
detection is stored, so an all-disabled file is rejected outright.

A rule scopes by `class`, `cameras`, `zones`, and `min_confidence`, then decides
delivery with `notify`, `severity`, `cooldown_seconds`, `notifications_table`,
and a `message` template with `{zone}` / `{camera}` placeholders.

Cooldown is keyed on `(rule, camera, zone)`, **not** on `track_id` — see
[DATA_MODEL](DATA_MODEL.md#detections-vs-notifications) for why that decides how
you count people.

### `notifications.yaml` — delivery only

Where an alert that **already fired** is delivered. It cannot cause or suppress
a rule; it routes what rules produced.

`log.enabled` is the local channel. Each webhook has its own `rules` and
`severities` filters, so one endpoint can take critical security alerts while
another takes everything — set by naming rules on the webhook, not by
duplicating rules.

The webhook path is live-only: never persisted, never retried. Losing it loses
live alerts, not stored rows.

### `collection.yaml` — dataset building

`output_dir` plus a list of `sessions`, each pinned to a camera with filters,
sampling, and an optional schedule. Frames are saved to disk for training data.

**The only file allowed to be entirely empty** — collection is optional and a
blank file means the feature is off. The file must still exist. With content,
`sessions: []` is the explicit "none" and is also accepted.

### Tracker config

`botsort_tracker.yaml` holds real values passed straight into boxmot's
`BotSort`, plus four `reid_*` keys this project consumes to build the ReID
model.

Two cautions. It is **not covered by strict validation**, and `BotSort` accepts
`**kwargs` — so `det_thresh`, `max_age`, `min_hits`, `iou_threshold`,
`asso_func` and `per_class` are absorbed and echoed in the startup log while
nothing reads them. They belong to other boxmot trackers. A typo is swallowed
the same way. Confirm a value arrived by reading the `BotSort: ...` startup
line, not by trusting the file.

And `frame_rate` must be the **measured** rate, not `fps_target` — it scales how
long a lost track is remembered. Details in
[ARCHITECTURE](ARCHITECTURE.md#tracking).

---

## Two conventions

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

**Two deliberate exceptions**, where a list is a collection rather than a filter
and being empty is a working state:

- `notifications.yaml` → `webhooks: []` — alerts go to the log only
- `collection.yaml` → `sessions: []` — no dataset collection

### `null` means "not set", and the key is still required

For scalars, an explicit `null` is the equivalent of `"*"`:

- `rules.yaml` → `min_confidence:` — no floor beyond the model's own
- `rules.yaml` → `notifications_table:` — webhook fires, nothing is stored
- `collection.yaml` → `schedule.after` / `before` / `start_date` / `end_date`
- `collection.yaml` → `sampling.interval_seconds` / `frames_per_minute`

The key must still be present. Absent is an oversight; `null` is a choice.

---

## Ownership

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

As a graph:

```
device.yaml ──────────────────────────► nodes rows
api.yaml ─────────────────────────────► where everything is sent

models.yaml ──id──► cameras.yaml ──id──► rules.yaml ──name──► notifications.yaml
     │                   │                    │
     │                   └──zone name─────────┘
     │
     └──tracker path──► botsort_tracker.yaml   (only when use_tracker: true)

collection.yaml ──camera id──► cameras.yaml
```

Arrows point from the reference to its definition. Nothing points backwards —
no file can redefine a value another file owns.

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
  Otherwise a detection outside every polygon is tagged `unzoned` — a real zone
  value that `rules.yaml` → `zones: "*"` still matches.

Draw them with the zone builder rather than by hand — it prints paste-ready
YAML at the camera's true resolution. See [TOOLS](TOOLS.md#zones-mode).

---

## Classes

- Names must match exactly what the model outputs — lowercase for COCO models
  (`person`, `car`, `truck`), or your training config's names for custom weights.
- `models.yaml` → `classes` is the source of truth. It takes no `"*"`: it is
  what camera filters are validated against, and resolving `"*"` would require
  loading the model before the config could be checked.
- Declare the classes you actually use, not the model's full list. Treat it as a
  contract — if the wrong weights file is deployed, the declared classes will
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
built for that exact GPU and TensorRT version. A `.pt` is the opposite:
compiled on an x86 machine and copied to the Pi. See
[ARCHITECTURE](ARCHITECTURE.md#model-formats).

---

## When a config is wrong

Every problem in a file is reported at once, so one restart shows the whole list
rather than one error per attempt:

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

Validate without starting the pipeline:

```bash
python3 -c "from core.config import load_config; c = load_config('config'); print('OK:', len(c.cameras), 'cameras')"
```

---

<div align="center">
<br/>

**[Docs index](README.md)** · **[Architecture](ARCHITECTURE.md)** · **[Data model](DATA_MODEL.md)** · **[Samples](../config/config_sample/)**

<br/>
</div>
