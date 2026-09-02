<div align="center">

```
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗  
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝  
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
```

### **Deployment Runbook**

<br/>

[![Platform](https://img.shields.io/badge/Jetson-Orin%20NX-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](https://developer.nvidia.com/embedded/jetson-orin)
[![Platform](https://img.shields.io/badge/Jetson-DeepStream-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](https://developer.nvidia.com/deepstream-sdk)
[![Backend](https://img.shields.io/badge/Postgres-per--branch%20schema-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](https://postgresql.org)

<br/>

> *Bringing up a new device, end to end — and the four places it silently goes wrong.*

</div>

---

## Order of Operations

The sequence is not arbitrary. Each stage produces something the next one needs.

```
1. BACKEND    create the branch              → branch_id + API key
2. DATABASE   provision the branch schema    → three tables + indexes
3. EDGE       install, export, configure     → device pushing rows
4. DASHBOARD  widgets over the real data     → analytics
```

The edge cannot be configured before stage 1, because `api.yaml` needs the branch
id and key. And it cannot successfully ingest before stage 2, because the tables
it writes into do not exist yet.

---

## Stage 1 — Create the Branch

Create the branch through the VisionEngine dashboard. This gives you two values
you will need on the device, and creates the branch's Postgres schema containing
**only** `dashboard_config`.

Record:

| Value | Where it goes |
|---|---|
| `branch_id` (UUID) | `api.yaml` → `branch_id` |
| API key (`cvp-…`) | `api.yaml` → `key` |
| Schema name (`<name>_branch_<hex>`) | Used in stage 2 |

---

## Stage 2 — Provision the Branch Schema

> **This step is manual and there is no error if you skip it.**
>
> Branch creation makes `dashboard_config` and stops. It does not create
> `detections`, `nodes`, or `notifications`. Skip this and the dashboard looks
> fine, while every ingest request fails with *relation does not exist* — the
> device buffers locally, then starts discarding rows once the buffer hits its
> size cap. Nothing surfaces until someone asks why the dashboard is empty.

Replace the schema name throughout, then run all of it.

### Tables

```sql
CREATE TABLE <schema>.nodes (
    id                BIGSERIAL PRIMARY KEY,
    device_id         TEXT        NOT NULL,
    name              TEXT        NOT NULL,
    location          TEXT        NOT NULL,
    status            TEXT        NOT NULL,
    cameras_active    INTEGER     NOT NULL,
    cameras_error     INTEGER     NOT NULL,
    detections_total  INTEGER     NOT NULL,
    buffer_pending    INTEGER     NOT NULL,
    uptime_seconds    NUMERIC     NOT NULL,
    ts                TIMESTAMPTZ NOT NULL
);

CREATE TABLE <schema>.detections (
    id            BIGSERIAL PRIMARY KEY,
    camera_id     TEXT,
    camera_name   TEXT,
    model_id      TEXT,
    track_id      TEXT,
    class         TEXT,
    confidence    NUMERIC,
    bbox_x1       INTEGER,  bbox_y1 INTEGER,
    bbox_x2       INTEGER,  bbox_y2 INTEGER,
    bbox_w        INTEGER,  bbox_h  INTEGER,
    anchor_x      NUMERIC,  anchor_y      NUMERIC,
    anchor_x_norm NUMERIC,  anchor_y_norm NUMERIC,
    frame_w       INTEGER,  frame_h INTEGER,
    zone          TEXT,
    ts            TIMESTAMPTZ
);

CREATE TABLE <schema>.notifications (
    id            BIGSERIAL PRIMARY KEY,
    rule_name     TEXT,
    severity      TEXT,
    message       TEXT,
    camera_id     TEXT,
    camera_name   TEXT,
    model_id      TEXT,
    track_id      TEXT,
    class         TEXT,
    confidence    NUMERIC,
    bbox_x1       INTEGER,  bbox_y1 INTEGER,
    bbox_x2       INTEGER,  bbox_y2 INTEGER,
    bbox_w        INTEGER,  bbox_h  INTEGER,
    anchor_x      NUMERIC,  anchor_y      NUMERIC,
    anchor_x_norm NUMERIC,  anchor_y_norm NUMERIC,
    frame_w       INTEGER,  frame_h INTEGER,
    zone          TEXT,
    ts            TIMESTAMPTZ
);
```

### Indexes

Not optional. A four-camera device writes on the order of a million rows a day,
and every dashboard widget sequentially scans without these — fine for a week,
then unusable.

```sql
CREATE INDEX detections_class_ts_idx ON <schema>.detections (class, ts DESC);
CREATE INDEX detections_zone_ts_idx  ON <schema>.detections (zone, ts DESC);
CREATE INDEX detections_track_idx    ON <schema>.detections (track_id, ts);
CREATE INDEX detections_ts_idx       ON <schema>.detections (ts DESC);

CREATE INDEX notifications_ts_idx    ON <schema>.notifications (ts DESC);
CREATE INDEX notifications_zone_idx  ON <schema>.notifications (camera_name, zone, ts DESC);

CREATE INDEX nodes_device_ts_idx     ON <schema>.nodes (device_id, ts DESC);

ANALYZE <schema>.detections;
ANALYZE <schema>.notifications;
ANALYZE <schema>.nodes;
```

`ANALYZE` matters: without fresh statistics the planner may keep choosing a
sequential scan and you will conclude the indexes did nothing.

### Verify

```sql
SELECT table_name FROM information_schema.tables WHERE table_schema = '<schema>';
-- expect: dashboard_config, detections, nodes, notifications

SELECT indexname FROM pg_indexes
WHERE schemaname = '<schema>' AND indexname NOT LIKE '%_pkey';
-- expect: 7 rows
```

---

## Stage 3 — Edge Device

### 3.1 Clone and install

```bash
sudo mkdir -p /opt/visionengine && sudo chown $USER:$USER /opt/visionengine
git clone https://github.com/WajeehAlamoudi/VisionEngine-Edge.git /opt/visionengine
cd /opt/visionengine && bash scripts/install.sh
```

The installer asks for the device type and sets up torch accordingly. On a
Jetson it detects the JetPack torch already present and creates the venv with
`--system-site-packages` so pip reuses it. It also copies all eight config
templates from `config/config_sample/`.

Confirm the GPU is actually visible from inside the venv before continuing:

```bash
source .venv/bin/activate
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You want `True`, and on a Jetson the torch path should be the system one under
`/home/<user>/.local/`, not inside `.venv`.

### 3.2 Detector model

Weights download automatically on first export. **Export on the target device** —
a TensorRT engine is built for that exact GPU and TensorRT version and will not
load anywhere else.

```bash
cd /opt/visionengine/models && source ../.venv/bin/activate
yolo export model=yolo26n.pt format=engine half=True imgsz=640 device=0
```

The build goes quiet for several minutes after `Local timing cache in use` — that
is kernel profiling, not a hang. Wait for `export success`.

> A `.pt` is the opposite case: it runs anywhere, which is why it stays the
> fallback when an engine will not build.

### 3.3 ReID model

Two steps, and **not** with `boxmot export` — see [Traps](#traps) for why.

**ONNX, with a dynamic batch dimension:**

```bash
cd /opt/visionengine && source .venv/bin/activate && python3 - <<'PY'
import torch
from pathlib import Path
from boxmot.reid.backends.pytorch_backend import PyTorchBackend

w   = Path('.venv/lib/python3.10/site-packages/models/osnet_x0_25_msmt17.pt')
out = w.with_suffix('.onnx')
model = PyTorchBackend(w, torch.device('cpu'), half=False).model.eval()

torch.onnx.export(
    model, torch.zeros(1, 3, 256, 128), str(out),
    input_names=['images'], output_names=['features'],
    opset_version=17, dynamo=False,
    dynamic_axes={'images': {0: 'batch'}, 'features': {0: 'batch'}},
)
print('wrote', out)
PY
```

**Engine, with a shape profile:**

```bash
cd /opt/visionengine/.venv/lib/python3.10/site-packages/models && \
/usr/src/tensorrt/bin/trtexec \
  --onnx=osnet_x0_25_msmt17.onnx \
  --saveEngine=osnet_x0_25_msmt17.engine \
  --fp16 \
  --minShapes=images:1x3x256x128 \
  --optShapes=images:4x3x256x128 \
  --maxShapes=images:16x3x256x128 \
  --memPoolSize=workspace:4096
```

`256x128` is OSNet's trained input size — do not change it. Only the leading
batch number varies, and it means *people in frame at once*: `opt` is the typical
count, `max` a hard ceiling that fails when exceeded.

`trtexec` prints a latency summary. Compare `GPU Compute Time mean` against the
PyTorch baseline to confirm the export was worth it.

### 3.4 Configure

All eight files need real values. The loader rejects the device if any key is
missing — that is deliberate. See [CONFIGURATION](CONFIGURATION.md) for
the full field reference.

| File | Needs |
|---|---|
| `api.yaml` | `branch_id` and `key` from stage 1, backend URL |
| `device.yaml` | Device id, location, log level, heartbeat |
| `models.yaml` | Engine path, `device: cuda`, classes, tracker config path |
| `cameras.yaml` | RTSP sources, `fps_target`, zones per camera |
| `rules.yaml` | What is stored and what alerts — **the storage gate** |
| `notifications.yaml` | Log channel, webhooks and their rule filters |
| `collection.yaml` | Dataset sessions, or empty |
| `botsort_tracker.yaml` | Tracker params, ReID backend and engine path |

For TensorRT ReID, `botsort_tracker.yaml` needs:

```yaml
reid_backend: tensorrt
reid_device: auto
reid_half: true
reid_weights: "/opt/visionengine/.venv/lib/python3.10/site-packages/models/osnet_x0_25_msmt17.engine"
```

Validate before running:

```bash
python3 -c "
from core.config import load_config
c = load_config('config')
print('OK:', len(c.cameras), 'cameras,', sum(len(x.zones) for x in c.cameras), 'zones')
"
```

Every problem in a file is reported together, so one pass fixes them all.

### 3.5 Run, then measure, then tune

```bash
python3 main.py
```

Startup should show, per camera:

```
detector '<id>': loading ./models/<engine> on cuda
ReID ready — tensorrt backend on cuda (half=True, weights=…engine)
camera 'cam-01': stream ready  960x480
```

Every 10 seconds each camera reports its measured rate:

```
camera 'cam-01': 10.4 fps (target 15) | 104 frames, 178 detections in 10s
```

**Now set `frame_rate`.** It cannot be guessed at configure time — it must match
the measured rate, because it scales how long a lost track is remembered:

```
memory in seconds = track_buffer / 30      (only while frame_rate is accurate)
```

Set `frame_rate` in `botsort_tracker.yaml` to the figure above and restart.

Install as a service once it is stable:

```bash
sudo bash scripts/service.sh install
sudo bash scripts/service.sh start
bash scripts/service.sh logs
```

---

## Stage 4 — Dashboard

Widgets live as rows in `<schema>.dashboard_config`. Each carries a `config`
JSONB with a `source` block the backend compiles into SQL against that branch's
own schema, plus a `refresh_seconds` column controlling auto-refresh
(`NULL` = never).

Columns available to widget queries:

| Table | Columns |
|---|---|
| `detections` | `camera_id` `camera_name` `model_id` `track_id` `class` `confidence` `bbox_*` `anchor_x/y` `anchor_x_norm/y_norm` `frame_w/h` `zone` `ts` |
| `notifications` | everything above, plus `rule_name` `severity` `message` |
| `nodes` | `device_id` `name` `location` `status` `cameras_active` `cameras_error` `detections_total` `buffer_pending` `uptime_seconds` `ts` |

### What makes a good widget here

Count **identities**, not detections. A person standing still for a minute
produces hundreds of rows; that number measures GPU effort, not footfall. The
metrics that mean something are built on `track_id` and `zone`:

```sql
COUNT(DISTINCT track_id)                            -- unique visitors
COUNT(DISTINCT track_id || '|' || zone)             -- zone visits
MAX(ts) - MIN(ts)  GROUP BY track_id, zone          -- dwell per person per zone
```

`track_id` is stable **within one camera** and independent across cameras — the
same person on two cameras gets two ids. Zone-level aggregates work across
cameras; individual journeys do not.

### Query engine limits

The widget `source` compiles to a single flat `SELECT`. Consequences:

- `from` must be a **bare table name** in that schema. Subqueries and views are rejected.
- One `GROUP BY` level only. Two-stage aggregations — the average of a per-person dwell — cannot be expressed.
- `SELECT`, `UNION`, and similar keywords are blocked in every fragment. This is an injection defence, not an oversight.
- `join` is supported for real tables, including a self-join on `detections`.

Set `refresh_seconds` per widget: short for live operations (nodes, alerts,
today's counts), long or `NULL` for anything spanning days. Each widget polls
independently, so twenty widgets at 5 s is four requests a second per open tab.

---

## Traps

Four failures that produce no error at the moment they happen.

### 1. `boxmot export` destroys a Jetson environment

boxmot checks for a pip package named `nvidia-tensorrt` before its TensorRT
export. On Jetson, TensorRT comes from JetPack and that package does not exist —
it only builds for x86_64. boxmot then runs `pip install`, which fails, but not
before replacing the JetPack torch with a generic wheel whose bundled CUDA the
driver cannot load. The device reports no CUDA at all, and the same install can
downgrade boxmot far enough to move modules this project imports.

**Use `torch.onnx.export` and `trtexec` instead.** Neither can install anything.

Recovery, if it has already happened:

```bash
pip uninstall -y torch torchvision           # reveals the system build again
pip install "boxmot>=19.0.0"                 # undo any downgrade
pip list --local | grep -E "nvidia|cuda-|triton"   # orphaned CUDA packages
```

### 2. A fixed batch-1 ReID engine

One crop is embedded per detection, so three people in frame is a batch of three.
An engine exported without `dynamic_axes` accepts only one and works perfectly in
testing with a single person. Always export with a shape profile.

### 3. Zone gaps

Zone membership for a person is tested at the **bottom-centre of the box** —
where the feet are. Draw polygons on the floor, not around a standing body.

Adjacent zones that do not actually touch leave a hole, and detections landing in
it are tagged `unzoned`. Every zone widget then reads empty while tracking works
perfectly. Check coverage against the frame dimensions before trusting the data.

Zones are evaluated in order and **first match wins**, so a frame-wide zone must
come last or it swallows everything after it.

### 4. `rules.yaml` gates storage, not just alerts

A detection matching no enabled rule is discarded entirely — it never reaches
`detections`. A class no rule mentions is invisible to the whole system.

And it inverts: with **zero** enabled rules the filter opens rather than closes,
and every detection of every class is stored. Config validation now rejects an
all-disabled file for this reason.

---

## Reference Benchmark

Jetson Orin NX, JetPack 6.2, TensorRT 10.7, four cameras at 960×480, BoT-SORT
with appearance matching enabled throughout. Measured over 10-second windows
with people in frame.

| Configuration | fps / camera | Total |
|---|---|---|
| `yolo26s` FP16 + PyTorch ReID | 3.4 – 3.6 | ~14 |
| `yolo26n` FP16 + PyTorch ReID | 4.0 – 4.7 | ~17 |
| **`yolo26n` FP16 + TensorRT ReID** | **9.6 – 10.7** | **~41** |

Per-stage frame budget, single camera:

| Stage | PyTorch ReID | TensorRT ReID |
|---|---|---|
| Detector inference | 22.18 ms | 22.18 ms |
| ReID forward pass | 43.92 ms | **1.37 ms** |
| Crop and resize | 0.68 ms | 0.68 ms |
| Track association | 0.39 ms | 0.39 ms |
| **Frame total** | **64.6 ms** | **24.6 ms** |

Two things this establishes, both counter-intuitive:

**INT8 on the detector gained nothing**, and neither did **FP16 on the ReID
model**. OSNet is small enough that its 44 ms was kernel launch overhead rather
than arithmetic — halving the arithmetic changes nothing, fusing the layers
changes everything.

**The bottleneck moved.** ReID fell from 65% of the frame to 6%; the detector
rose from 34% to 90%. Detector-side optimisations that were worthless before —
INT8, a smaller `input_size` — are now the ones worth testing.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `cuda avail: False` after any pip command | A generic torch wheel shadowing the vendor build. Uninstall it from the venv |
| `Config error in <file>` at startup | A key is missing, mistyped, or out of range. The message names the field |
| `failed to open source` on some cameras | NVR concurrent-stream limit, not a pipeline fault |
| Dashboard empty, device looks healthy | Branch schema missing its tables — stage 2 was skipped |
| Every alert says `unzoned` | Zone polygons do not cover where people stand |
| Track ids change constantly | `frame_rate` does not match the measured rate |
| Widgets show an amber warning | The widget's SQL was rejected — check for a subquery in `from` |
| FPS falls as the day goes on | Missing indexes; the tables have grown |

---

<div align="center">
<br/>

**[Docs index](README.md)** · **[Configuration](CONFIGURATION.md)** · **[Architecture](ARCHITECTURE.md)** · **[Data model](DATA_MODEL.md)** · **[Tools](TOOLS.md)**

<br/>
</div>
