<div align="center">

```
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗  
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝  
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
```

### **Edge-native computer vision pipeline. Detect. Track. Ingest.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-1a1a2e?style=for-the-badge&logo=python&logoColor=4fc3f7)](https://python.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-1a1a2e?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iIzAwZmZmZiIgZD0iTTEyIDJMMiA3bDEwIDUgMTAtNXoiLz48L3N2Zz4=&logoColor=4fc3f7)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RPi%20%7C%20Jetson%20%7C%20Mac-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](https://github.com/WajeehAlamoudi/VisionEngine-Edge)

<br/>

> *Runs on a Raspberry Pi. Streams to Postgres. Zero cloud dependency on the edge.*

</div>

---

## What It Does

VisionEngine Edge is the on-device half of the VisionEngine platform. It runs on any Linux box with a camera — from a Raspberry Pi 5 to a Jetson Orin — and does one job exceptionally well:

**Frame in → Detections out → Database.**

Every detected object is enriched with spatial context, matched against configurable rules, and pushed to your branch schema in PostgreSQL via a local SQLite buffer that survives network outages. No frames leave the device. No cloud inference. No vendor lock-in.

---

## Pipeline

```
                    ┌─────────────────────────────────────────────────────┐
                    │                  EDGE DEVICE                        │
                    │                                                     │
  Camera Stream ───►│  cap.read()                                         │
  USB / RTSP        │      │                                              │
                    │   FPS throttle                                      │
                    │      │                                              │
                    │   ModelRunner.run()          BoT-SORT tracker       │
                    │   YOLO predict / track  ◄──  (optional, per model)  │
                    │      │                                              │
                    │   enrich()                                          │
                    │   zone tag · anchor · normalize                     │
                    │      │                                              │
                    │   RulesEngine.filter_and_tag()                      │
                    │      │                                              │
                    │   ┌──┴──────────────┐                              │
                    │   │  no rule match  │──► discard                   │
                    │   └─────────────────┘                              │
                    │      │ matched                                      │
                    │      │                                              │
                    │   detection_row()    notification_row()             │
                    │      │                     │                        │
                    │      └──────────┬──────────┘                       │
                    │                 │                                   │
                    │           buffer.write()                            │
                    │           SQLite  [offline-resilient]               │
                    │                 │                                   │
                    │           IngestWorker                              │
                    │           POST /branches/ingest                     │
                    └─────────────────┼───────────────────────────────────┘
                                      │  HTTP + X-API-Key
                                      ▼
                              ┌───────────────┐
                              │  PostgreSQL   │
                              │               │
                              │  detections   │
                              │  notifications│
                              │  nodes        │
                              └───────────────┘
```

---

## Features

- **YOLO inference** via [ultralytics](https://ultralytics.com) — any `.pt`, `.onnx`, `.tflite`, `.hef` model
- **BoT-SORT tracking** — persistent `track_id` per object across frames, per-model toggle
- **Zone detection** — polygon zones in native camera resolution, click-to-draw debug tool
- **Rules engine** — filter, notify, cooldown — per class, per camera, per zone
- **Offline buffer** — SQLite WAL, survives network outages, exponential backoff on retry
- **Heartbeat** — device health rows pushed to `nodes` table on a configurable interval
- **Dataset collection** — frame sampler with schedule, filters, and save modes
- **Debug tool** — three live modes: view stream, draw zones, run inference overlay
- **Hardware agnostic** — CPU, CUDA, and Apple MPS all work with a single `device:` change in `models.yaml`; Hailo runs through its own dedicated backend (see [Hailo Backend](#hailo-backend) below — more involved than a config change)

---

## Hardware

| Device | Inference | Recommended FPS | Status |
|--------|-----------|-----------------|--------|
| Raspberry Pi 4 (CPU) | PyTorch `.pt` | 1–2 | Verified |
| Raspberry Pi 5 (CPU) | PyTorch `.pt` | 2–3 | Verified |
| Raspberry Pi + Hailo-8/8L | Hailo `.hef` | TBD | Blocked — see [Hailo Backend](#hailo-backend) |
| Jetson Nano | CUDA `.pt` | 5–8 | Estimated |
| Jetson Orin | CUDA `.pt` | 15–30 | Estimated |
| Mac Mini (M-series) | MPS / CoreML | 10–20 | Estimated |

FPS figures not marked "Verified" are estimates, not measurements — treat them as a starting expectation, not a spec.

---

## Hailo Backend

CPU, CUDA, and MPS all run through the same Ultralytics/torch code path — swapping between them is genuinely a one-line `device:` change in `models.yaml`. Hailo does not work this way, and that's not a temporary gap — it's a structural difference worth understanding before planning around it.

### Why Hailo is different

- A `.hef` file has no PyTorch/Ultralytics runtime behind it at all. Inference on a Hailo chip runs entirely through **HailoRT** (Hailo's own SDK), via a separate `HailoDetector` backend (`core/model/detector/hailo_detector.py`) — not through `ultralytics.YOLO.predict()`.
- Getting a model onto Hailo is a two-step, two-machine process:
  1. **Convert** `.pt → .hef` using Ultralytics' export (`yolo export model=X.pt format=hailo name=hailo8`) plus Hailo's Dataflow Compiler. This step only runs on **x86_64 Linux with Python 3.10** — not on the edge device itself (Hailo compilation isn't supported on ARM), and not on any other Python version (the Dataflow Compiler is only built for 3.10).
  2. **Deploy** the resulting `.hef` *and* its accompanying `metadata.yaml` (written alongside it by the export) together to the edge device — `HailoDetector` reads the model's real class list from `metadata.yaml` at load time, since a `.hef` carries no class names of its own. Copying the `.hef` alone is not sufficient.
- Only **YOLOv8, YOLO11, and YOLO26** architectures can export to Hailo format today. Other YOLO versions (e.g. YOLOv12) are rejected by Ultralytics' own exporter before it even reaches the Hailo compiler — this is not a configuration problem and can't be worked around from this codebase.

### Known limitation — dependency conflict on-device

HailoRT versions **before 5.3.0** require `numpy==1.23.3` to run inference correctly; without it, `hailo_platform` returns a zero-byte input buffer to the chip regardless of what data is actually sent, and every inference call fails with:
```
[HailoRT] [error] CHECK failed - Memory size of vstream ... does not match the frame count! (Expected N, got 0)
```
This is not fixable from application code — it's a numpy ABI compatibility issue inside HailoRT's compiled bindings.

The conflict: the tracker library this project uses (`boxmot`) requires `numpy>=2.2.0` — the exact opposite constraint. The two cannot be pinned to a mutually compatible numpy version in one shared Python environment.

As of this writing, **Raspberry Pi OS's own apt repository only packages HailoRT up to 4.20.0** — the numpy-2.x-compatible 5.3.0+ release isn't available through it. Installing a newer HailoRT manually is possible but uses a driver/firmware stack Raspberry Pi hasn't packaged or tested for their AI Kit integration, and carries real risk to a working setup.

**Net effect:** on a device where both Hailo detection and BoT-SORT tracking (`use_tracker: true`) need to run together in the current single-process architecture, this dependency conflict currently blocks it from running end-to-end.

### Next planned change

Isolate `HailoDetector`'s inference call into its own dedicated environment (a separate Python 3.11 venv pinned to `numpy==1.23.3`, containing only `hailo_platform`) and have it communicate with the main process — instead of importing `hailo_platform` directly into the same process as `boxmot`/`torch`. This keeps HailoRT's dependency constraints from ever touching the rest of the stack, without waiting on an upstream HailoRT release or an apt package update. Not yet implemented.

---

## Quick Start

### 1 — Clone

```bash
git clone https://github.com/WajeehAlamoudi/VisionEngine-Edge.git /opt/visionengine
cd /opt/visionengine
```

### 2 — Install

```bash
python -m venv venv && source venv/bin/activate

# Standard (CPU / CUDA / MPS)
pip install -r requirements.txt --timeout 300

# Raspberry Pi — CPU-only torch (saves ~1 GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --timeout 300
```

### 3 — Configure

Copy the sample files and fill them in:

```bash
cp config/config_sample/api.sample.yaml          config/api.yaml
cp config/config_sample/cameras.sample.yaml      config/cameras.yaml
cp config/config_sample/device.sample.yaml       config/device.yaml
cp config/config_sample/models.sample.yaml       config/models.yaml
cp config/config_sample/notifications.sample.yaml config/notifications.yaml
cp config/config_sample/rules.sample.yaml        config/rules.yaml
```

### 4 — Run

```bash
python main.py
# custom config directory
python main.py --config /etc/visionengine
```

---

## Configuration

Seven files. Each has one responsibility.

```
config/
├── api.yaml            ← branch_id, API key, ingest batch settings
├── cameras.yaml        ← camera sources, model binding, zones, routing
├── device.yaml         ← device identity, FPS, tracker, heartbeat, buffer
├── models.yaml         ← model paths, classes, confidence, tracker toggle
├── rules.yaml          ← filter rules, notification rules, cooldowns
├── notifications.yaml  ← webhook targets (VisionEngine, Slack, Teams, custom)
├── collection.yaml     ← dataset collection sessions (optional)
└── config_sample/      ← fully-commented reference for every field
    ├── api.sample.yaml
    ├── cameras.sample.yaml
    ├── device.sample.yaml
    ├── models.sample.yaml
    ├── notifications.sample.yaml
    ├── rules.sample.yaml
    └── collection.sample.yaml
```

### Default 3-Table Setup

The default configuration writes to exactly three tables in your branch schema:

| Table | Written by | Content |
|-------|-----------|---------|
| `detections` | `detection_row()` | One row per detection — all classes, all cameras |
| `notifications` | `notification_row()` | One row per alert fired by a rule |
| `nodes` | `_heartbeat_row()` | Device health pulse on a configurable interval |

Table names are set in YAML — change them at any time without touching code.

### BoT-SORT Tracker

Enable per model in `models.yaml`. All cameras using that model get persistent `track_id` values — no per-camera config needed.

```yaml
# models.yaml
models:
  - id: general_yolo12n
    path: ./models/yolov12n.pt
    use_tracker: true       # ← flip this ON
    ...
```

```yaml
# device.yaml
device:
  tracker: "botsort.yaml"   # botsort.yaml | bytetrack.yaml | custom path
```

Cameras sharing a model with `use_tracker: false` share one model instance (RAM efficient). Cameras with `use_tracker: true` each get a dedicated instance — tracker state is per-camera.

---

## Database Schema

```sql
-- All detections from all cameras and classes
CREATE TABLE <schema>.detections (
    id            BIGSERIAL PRIMARY KEY,
    camera_id     TEXT,          camera_name  TEXT,
    model_id      TEXT,          track_id     TEXT,
    class         TEXT,          confidence   NUMERIC,
    bbox_x1       INTEGER,       bbox_y1      INTEGER,
    bbox_x2       INTEGER,       bbox_y2      INTEGER,
    bbox_w        INTEGER,       bbox_h       INTEGER,
    anchor_x      NUMERIC,       anchor_y     NUMERIC,
    anchor_x_norm NUMERIC,       anchor_y_norm NUMERIC,
    frame_w       INTEGER,       frame_h      INTEGER,
    zone          TEXT,          ts           TIMESTAMPTZ
);

-- One row per alert fired by a rule
CREATE TABLE <schema>.notifications (
    id            BIGSERIAL PRIMARY KEY,
    rule_name     TEXT,          severity     TEXT,
    message       TEXT,
    camera_id     TEXT,          camera_name  TEXT,
    model_id      TEXT,          track_id     TEXT,
    class         TEXT,          confidence   NUMERIC,
    bbox_x1       INTEGER,       bbox_y1      INTEGER,
    bbox_x2       INTEGER,       bbox_y2      INTEGER,
    bbox_w        INTEGER,       bbox_h       INTEGER,
    anchor_x      NUMERIC,       anchor_y     NUMERIC,
    anchor_x_norm NUMERIC,       anchor_y_norm NUMERIC,
    frame_w       INTEGER,       frame_h      INTEGER,
    zone          TEXT,          ts           TIMESTAMPTZ
);

-- Device health — one row per heartbeat interval
CREATE TABLE <schema>.nodes (
    id               BIGSERIAL PRIMARY KEY,
    device_id        TEXT,        name             TEXT,
    location         TEXT,        status           TEXT,
    cameras_active   INTEGER,     cameras_error    INTEGER,
    detections_total INTEGER,     buffer_pending   INTEGER,
    uptime_seconds   NUMERIC,     ts               TIMESTAMPTZ
);
```

<details>
<summary><strong>SQL analytics you can run immediately</strong></summary>

```sql
-- Detections per zone per minute
SELECT zone, date_trunc('minute', ts) AS minute, COUNT(*) AS count
FROM detections WHERE class = 'person'
GROUP BY zone, minute ORDER BY minute;

-- Dwell — track a person's journey across zones
SELECT track_id, zone, MIN(ts) AS entered, MAX(ts) AS last_seen,
       EXTRACT(EPOCH FROM MAX(ts) - MIN(ts)) AS dwell_seconds
FROM detections WHERE class = 'person' AND track_id IS NOT NULL
GROUP BY track_id, zone ORDER BY entered;

-- Alert frequency by rule and camera
SELECT rule_name, camera_id, COUNT(*) AS alerts,
       MAX(ts) AS last_alert
FROM notifications
GROUP BY rule_name, camera_id ORDER BY alerts DESC;

-- Object size distribution (estimate distance from camera)
SELECT class, AVG(bbox_h) AS avg_height_px,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY bbox_h) AS median_height_px
FROM detections GROUP BY class;

-- Device health — last seen per node
SELECT device_id, name, location, status, cameras_active,
       buffer_pending, ts AS last_heartbeat
FROM nodes WHERE ts = (SELECT MAX(ts) FROM nodes n2 WHERE n2.device_id = nodes.device_id);
```

</details>

---

## Debug Tool

Three modes for on-device diagnostics. No pipeline running required.

```bash
# View live stream — check resolution, FPS, connectivity
python tools/debug.py --mode view --source 0
python tools/debug.py --mode view --source "rtsp://admin:pass@192.168.1.100/stream1"

# Draw zones interactively — click polygon points, press N to close,
# S to print YAML output ready to paste into cameras.yaml
python tools/debug.py --mode zones --source 0
python tools/debug.py --mode zones --camera cam-01 --config /etc/visionengine

# Live inference overlay — bounding boxes, confidence, zones, FPS
python tools/debug.py --mode inference --camera cam-01 --config /etc/visionengine
```

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `N` | Close current zone polygon |
| `U` | Undo last point |
| `S` | Print zone YAML to terminal |
| `Z` | Toggle existing zones overlay |
| `D` | Toggle detection boxes (inference mode) |

> Zone coordinates are always captured in native camera resolution — the pipeline handles all resizing internally.

---

## Rules

Rules are the pipeline's gate. A detection that matches no rule is discarded before it reaches the buffer.

```yaml
# rules.yaml
rules:
  # Filter-only — detection stored silently, no webhook
  - name: track_persons_lobby
    class: person
    cameras: [cam-01]
    zones: [lobby_entrance]
    min_confidence: 0.75
    notify: false
    enabled: true

  # Notification rule — detection stored + webhook fired
  - name: person_restricted_area
    class: person
    cameras: []          # [] = all cameras
    zones: [server_room]
    min_confidence: 0.88
    cooldown_seconds: 15
    notify: true
    severity: critical
    notifications_table: "notifications"
    message: "Person in {zone} on {camera} — confidence {confidence}"
    enabled: true
```

**Rule evaluation:**

```
filter_and_tag() returns:
  None        → no rule matched → detection discarded
  []          → matched, notify: false → stored in detections, no webhook
  [RuleMatch] → matched, notify: true  → stored in detections + notifications, webhook fired
```

---

## Project Structure

```
VisionEngine-Edge/
├── main.py                     ← entry point
├── requirements.txt
├── config/                     ← deployment config (fill in, never commit secrets)
│   ├── api.yaml
│   ├── cameras.yaml
│   ├── device.yaml
│   ├── models.yaml
│   ├── notifications.yaml
│   ├── rules.yaml
│   ├── collection.yaml
│   └── config_sample/          ← fully-documented reference files
├── core/
│   ├── config/                 ← YAML parsers + validation + AppConfig
│   ├── pipeline/               ← per-camera pipeline loop + rows
│   ├── model/                  ← ModelRunner (predict/track) + ModelRegistry
│   ├── rules/                  ← RulesEngine + DetectionEvent + RuleMatch
│   ├── buffer/                 ← SQLite offline buffer (aiosqlite)
│   ├── ingest/                 ← IngestWorker — HTTP flush loop
│   ├── notifier/               ← webhook delivery + payload builder
│   ├── health/                 ← heartbeat rows + health file writer
│   └── collection/             ← dataset frame sampler
├── tools/
│   └── debug/                  ← view / zones / inference debug modes
└── models/                     ← model weight files (.pt, .onnx, .hef)
```

---

## Deployment

<details>
<summary><strong>Raspberry Pi setup (full walkthrough)</strong></summary>

```bash
# 1. Install system dependencies
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv libopencv-dev

# 2. Create install directory
sudo mkdir -p /opt/visionengine
sudo chown -R $USER:$USER /opt/visionengine

# 3. Clone
git clone https://github.com/WajeehAlamoudi/VisionEngine-Edge.git /opt/visionengine
cd /opt/visionengine

# 4. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Install (CPU-only torch saves ~1 GB on the Pi)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --timeout 300

# 6. Download a model
mkdir -p models
# place your .pt or .hef file in ./models/

# 7. Fill config
nano config/api.yaml       # branch_id + key + url
nano config/cameras.yaml   # camera source + model_id
nano config/device.yaml    # device id + name + location
nano config/models.yaml    # model path + classes
nano config/rules.yaml     # at least one rule

# 8. Run
python main.py
```

</details>

<details>
<summary><strong>Run as a systemd service</strong></summary>

```ini
# /etc/systemd/system/visionengine-edge.service
[Unit]
Description=VisionEngine Edge Pipeline
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/opt/visionengine
ExecStart=/opt/visionengine/venv/bin/python main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable visionengine-edge
sudo systemctl start visionengine-edge
sudo journalctl -u visionengine-edge -f
```

</details>

---

<div align="center">

**VisionEngine Edge** · Built for the edge. Designed to survive.

</div>
