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

## Documentation

This README is the front door. The full guides live in **[`docs/`](docs/)**.

| Guide | Answers |
|---|---|
| **[Architecture](docs/ARCHITECTURE.md)** | How the pipeline works — backends, TensorRT, tracking, ReID, zones, benchmarks |
| **[Configuration](docs/CONFIGURATION.md)** | What each of the 8 YAML files owns, and how they reference each other |
| **[Data Model](docs/DATA_MODEL.md)** | Every column of `detections`, `notifications`, `nodes` — and what it means |
| **[Deployment](docs/DEPLOYMENT.md)** | Backend → database → edge → dashboard, end to end |
| **[Tools](docs/TOOLS.md)** | Running the agent, the debug tool, the service, reading the logs |

---

## Pipeline

```
                    ┌─────────────────────────────────────────────────────┐
                    │                  EDGE DEVICE                        │
                    │                                                     │
  Camera Stream ───►│  CameraRuntime.read()  ── timestamped here          │
  USB / RTSP        │      capture + inference, matched to each other:    │
                    │      OpenCV → YOLO + BoT-SORT, or                   │
                    │      NVDEC → nvinfer → NvTracker (stays on the GPU) │
                    │      │                                              │
                    │   enrich()                                          │
                    │   zone tag · anchor · normalize                     │
                    │      │                                              │
                    │   RulesEngine.filter_and_tag()                      │
                    │      │                                              │
                    │   ┌──┴──────────────┐                               │
                    │   │  no rule match  │──► discard                    │
                    │   └─────────────────┘                               │
                    │      │ matched                                      │
                    │      │                                              │
                    │   detection_row()    notification_row()             │
                    │      │                     │                        │
                    │      └──────────┬──────────┘                        │
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

One pipeline per camera, each an independent task — a failing camera does not stop the others. Walkthrough in [Architecture](docs/ARCHITECTURE.md#the-pipeline).

---

## Features

- **Detection backends** — Ultralytics (`.pt`, `.engine`, `.mlpackage`) and DeepStream (`.engine`), each importing its SDK lazily so one device's runtime can't break another's
- **BoT-SORT tracking** with **OSNet ReID** — persistent `track_id` across frames and occlusion, per-model toggle
- **TensorRT acceleration** — detector and ReID both run as compiled engines; see the [benchmark](docs/ARCHITECTURE.md#benchmark)
- **Zone analytics** — polygon zones in native camera resolution, click-to-draw builder, person membership tested at the feet
- **Rules engine** — the gate for storage *and* alerts, per class, camera, zone, with cooldown
- **Strict configuration** — no defaults anywhere; a missing or out-of-range key stops startup with the file, field, and expectation named
- **Offline buffer** — SQLite WAL, survives network outages, replays with original capture timestamps
- **Heartbeat** — device health rows pushed to `nodes`, plus an optional local health file
- **Dataset collection** — frame sampler with schedule, filters, and save modes
- **Debug tool** — three live modes: view stream, draw zones, run inference overlay

---

## Hardware

| Device | Inference | FPS / camera | Status |
|--------|-----------|--------------|--------|
| Raspberry Pi 4 (CPU) | PyTorch `.pt` | 1–2 | Verified |
| Raspberry Pi 5 (CPU) | PyTorch `.pt` | 2–3 | Verified |
| **Jetson Orin NX** | **TensorRT `.engine` + TensorRT ReID** | **9.6–10.7** | **Verified** — 4 cameras @ 960×480, ~41 fps aggregate |
| Jetson Nano | CUDA `.pt` | 5–8 | Estimated |
| Mac Mini (M-series) | MPS / CoreML | 10–20 | Estimated |

The Jetson figures are measured with tracking and appearance matching **on** throughout, over 10-second windows with people in frame. The same hardware managed 4.0–4.7 fps/camera before the ReID model was moved to TensorRT — full per-stage breakdown in [Architecture § Benchmark](docs/ARCHITECTURE.md#benchmark).

Figures marked "Estimated" are not measurements — treat them as a starting expectation, not a spec.

---

## Quick Start

Standing up a new device for real — including the branch, the database schema, and the model exports — is [Deployment](docs/DEPLOYMENT.md). This is the short path.

### 1 — Clone

```bash
sudo mkdir -p /opt/visionengine && sudo chown -R $USER:$USER /opt/visionengine
git clone https://github.com/WajeehAlamoudi/VisionEngine-Edge.git /opt/visionengine
cd /opt/visionengine
```

### 2 — Install

```bash
bash scripts/install.sh
```

One interactive script, one entry point for every supported device. It asks what hardware this is (Raspberry Pi / CPU, Jetson, generic CUDA PC, Mac, Jetson + DeepStream) and sets up torch correctly for that device — including reusing an already-working Jetson torch instead of shadowing it with a broken generic build, which a plain `pip install` would do silently. It creates `.venv`, installs `requirements.txt`, creates `models/`, `data/`, `logs/`, `collected/`, and copies **all eight** config templates from `config/config_sample/` into `config/`.

It copies templates only — it never generates config content. Filling in real values is a deliberate, reviewable act.

### 3 — Configure

Every one of the eight files needs real values before the device will start. There are **no defaults**: a missing, mistyped, or out-of-range key stops startup with the file and field named.

Each field is documented inline in the matching `.sample.yaml`. What each *file* owns, and how they reference each other, is in [Configuration](docs/CONFIGURATION.md).

Check your work without starting the pipeline:

```bash
python3 -c "from core.config import load_config; c = load_config('config'); print('OK:', len(c.cameras), 'cameras')"
```

### 4 — Model weights

Not included in the repo (gitignored — they're large binaries). Place your `.pt` / `.engine` / `.mlpackage` file in `models/`, matching the `path:` you set in `models.yaml`.

A TensorRT `.engine` must be exported **on** the device that will run it. Both the detector and the ReID exports are in [Deployment § Stage 3](docs/DEPLOYMENT.md#stage-3--edge-device).

### 5 — Run

```bash
source .venv/bin/activate
python main.py
# custom config directory
python main.py --config /etc/visionengine
```

Every 10 seconds each camera logs its measured throughput. Read that before tuning anything — see [Tools § Reading the logs](docs/TOOLS.md#reading-the-logs).

Or run it as a background service — see [Deployment](#deployment) below.

---

## Configuration

Eight files, each with one responsibility.

```
config/
├── api.yaml              ← branch_id, API key, ingest batching, offline buffer
├── device.yaml           ← device identity, log level, heartbeat, health file
├── cameras.yaml          ← camera sources, model binding, classes, zones, routing
├── models.yaml           ← model paths, devices, classes, thresholds, tracker toggle
├── rules.yaml            ← what is stored, and what raises an alert
├── notifications.yaml    ← log channel and webhook delivery, per-rule routing
├── collection.yaml       ← dataset collection sessions (optional)
├── botsort_tracker.yaml  ← tracker tuning + ReID backend, device-specific
└── config_sample/        ← fully-commented reference for every field
```

`botsort_tracker.yaml` is the odd one out: it is read at model load time from the path in `models.yaml` → `tracker`, only when `use_tracker: true`, and is **not** covered by strict validation.

Two conventions run through all of them:

- **`"*"` means all.** An empty list is rejected wherever it would be ambiguous — `"*"` is a decision you wrote, `[]` is usually a deleted last entry.
- **`null` means "not set", and the key is still required.** Absent is an oversight; `null` is a choice.

**`rules.yaml` gates storage, not just alerts.** A detection matching no enabled rule is discarded before the buffer — so a class no rule mentions never reaches the database. Full detail in [Configuration](docs/CONFIGURATION.md).

### Tracking

Enable per model in `models.yaml`. Every camera using that model gets persistent `track_id` values.

```yaml
# models.yaml
models:
  - id: general_coco
    path: ./models/yolo26n.engine
    runtime: ultralytics
    accelerator: cuda
    use_tracker: true                        # ← flip this ON
    tracker: "config/botsort_tracker.yaml"   # ← tracker params live here
```

Cameras sharing a model with `use_tracker: false` share one model instance (RAM efficient). Cameras with `use_tracker: true` each get a dedicated instance — tracker state is per-camera.

One value in `botsort_tracker.yaml` deserves attention: `frame_rate` must be the camera's **measured** rate, not `fps_target`. It scales how long a lost track is remembered, and a stale value silently breaks identity in whichever direction is worse. See [Architecture § Tracking](docs/ARCHITECTURE.md#tracking).

---

## Database

Three tables per branch schema, written entirely by the edge:

| Table | Written by | Content |
|-------|-----------|---------|
| `detections` | `detection_row()` | One row per detected object per frame that matched a rule |
| `notifications` | `notification_row()` | One row per alert fired by a rule |
| `nodes` | `_heartbeat_row()` | Device health pulse on a configurable interval |

Table names come from config — `raw_table` and `routing[].table` in `cameras.yaml`, `notifications_table` per rule — so one camera can split people and vehicles into separate tables.

**They are not created automatically.** Creating a branch provisions only `dashboard_config`; the three data tables and their indexes are a manual step, and skipping it produces no error — the dashboard just stays empty. The DDL is in [Deployment § Stage 2](docs/DEPLOYMENT.md#stage-2--provision-the-branch-schema).

Every column, what it means, and how to aggregate it correctly is in [Data Model](docs/DATA_MODEL.md). The short version of the most common mistake:

```sql
COUNT(DISTINCT track_id) FROM detections     -- counts people
COUNT(*)                 FROM notifications  -- counts alerts, not people
```

---

## Debug Tool

Three modes for on-device diagnostics. No running pipeline required.

```bash
python tools/debug.py --mode view      --source 0                 # stream, resolution, FPS
python tools/debug.py --mode zones     --source 0                 # draw polygons, print YAML
python tools/debug.py --mode inference --camera cam-01            # live detection overlay
```

Keys, flags, and what each mode does and doesn't exercise: [Tools](docs/TOOLS.md#the-debug-tool).

> Zone coordinates are always captured in native camera resolution — the pipeline handles all resizing internally.

---

## Project Structure

```
VisionEngine-Edge/
├── main.py                     ← entry point
├── requirements.txt
├── docs/                       ← architecture, configuration, data model, deployment, tools
├── scripts/
│   ├── install.sh              ← interactive setup — one entry point for every device
│   └── service.sh              ← install/start/stop/logs as a systemd service
├── config/                     ← deployment config (fill in, never commit secrets)
│   ├── *.yaml                  ← the eight files listed above
│   └── config_sample/          ← fully-documented reference files
├── core/
│   ├── config/                 ← strict YAML parsers + cross-file validation + AppConfig
│   ├── pipeline/               ← per-camera loop, enrichment, row builders
│   ├── model/
│   │   ├── detector/           ← one package per runtime + the registry
│   │   │   ├── ultralytics/    ← YOLO detector + OpenCV capture
│   │   │   └── deepstream/     ← nvinfer parsing + the GStreamer pipeline
│   │   └── tracker/            ← BoT-SORT via boxmot, ReID backend selection
│   ├── zone/                   ← point-in-polygon zone assignment
│   ├── rules/                  ← RulesEngine + DetectionEvent + RuleMatch
│   ├── buffer/                 ← SQLite offline buffer (aiosqlite)
│   ├── ingest/                 ← IngestWorker — HTTP flush loop
│   ├── notifier/               ← webhook delivery + payload builder
│   ├── health/                 ← heartbeat rows + health file writer
│   └── collector/              ← dataset frame sampler
├── tools/
│   └── debug/                  ← view / zones / inference debug modes
└── models/                     ← model weight files (.pt, .engine, .mlpackage)
```

---

## Deployment

Standing up a device end to end — branch, schema, model exports, config, dashboard — is [**docs/DEPLOYMENT.md**](docs/DEPLOYMENT.md). It also documents the four failures that produce no error at the moment they happen.

To run as a background service instead of a manual terminal session, `scripts/service.sh` manages a `systemd` unit for you — no unit file to write or `.venv` path to get right by hand:

```bash
sudo bash scripts/service.sh install    # create and enable the service (run once)
sudo bash scripts/service.sh start      # start it
     bash scripts/service.sh status     # check it's running
     bash scripts/service.sh logs       # tail live logs (Ctrl+C to exit)
sudo bash scripts/service.sh restart    # restart after a config change
sudo bash scripts/service.sh uninstall  # remove the service — config and data are kept
```

The service runs as whichever user ran `install`, restarts automatically on failure, and starts on boot. Nothing is re-read at runtime — restart after any config change.

---

<div align="center">

**VisionEngine Edge** · Built for the edge. Designed to survive.

</div>
