<div align="center">

```
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗  
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝  
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
```

### **Tools & Operations**

<br/>

[![Entry](https://img.shields.io/badge/Run-main.py-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#running-the-agent)
[![Debug](https://img.shields.io/badge/Debug-3%20modes-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#the-debug-tool)
[![Service](https://img.shields.io/badge/Service-systemd-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#running-as-a-service)

<br/>

> *Everything you type at a device, and how to read what it says back.*

</div>

---

## Running the agent

```bash
python3 main.py
```

| Flag | Default | Purpose |
|---|---|---|
| `--config DIR` | `config` | Config directory to load |

That is the whole interface. Everything else is configuration — deliberately, so
a running device's behaviour is fully described by files under version control
rather than by how someone launched it.

Startup order is: load and validate all config → open buffer → load models →
build rules → start ingest, notifier, collector → start one pipeline per enabled
camera → start the heartbeat. A config problem stops the process before any
model loads, so a bad config costs seconds, not minutes.

Shutdown on `SIGINT`/`SIGTERM` is graceful — pipelines stop, then background
services flush in order. Give it a moment rather than sending a second signal;
`SIGKILL` leaves buffered rows unflushed.

### Validate config without running

```bash
python3 -c "from core.config import load_config; c = load_config('config'); print('OK:', len(c.cameras), 'cameras,', sum(len(x.zones) for x in c.cameras), 'zones')"
```

Every problem in a file is reported together, so one pass fixes them all.

---

## The debug tool

Three modes for configuring and verifying a deployment before the pipeline runs.

```bash
python3 tools/debug.py --mode <view|zones|inference> [options]
```

| Flag | Default | Applies to | Purpose |
|---|---|---|---|
| `--mode` | *required* | all | `view` \| `zones` \| `inference` |
| `--config DIR` | `config` | all | Config directory |
| `--camera ID` | — | all | Camera id from `cameras.yaml`. **Required** for `inference` |
| `--source SRC` | — | view, zones | Override the source: integer index, `rtsp://…`, or file path |

`--source` and `--camera` are two ways to name a stream. `--source` bypasses
config entirely, which is what you want before `cameras.yaml` exists.
`--camera` reads the source from config — and for `zones`, also loads existing
polygons so you can extend them.

### `view` mode

Confirms a stream is reachable and reports its true resolution and frame rate.

```bash
python3 tools/debug.py --mode view --source 0
python3 tools/debug.py --mode view --source "rtsp://192.168.1.50/stream1"
python3 tools/debug.py --mode view --camera cam-01
```

Run this **first** on any new camera. Resolution here is what zone coordinates
must be drawn against, and an RTSP URL that fails here will fail identically in
the pipeline — with a much noisier log.

`Q` quits.

### `zones` mode

Interactive polygon builder. Grabs one frame, closes the stream, and lets you
draw on the still — so a moving scene does not fight you.

```bash
python3 tools/debug.py --mode zones --source 0
python3 tools/debug.py --mode zones --camera cam-01 --config config/   # loads existing zones
```

| Key | Action |
|---|---|
| **left-click** | Add a point |
| `U` | Undo the last point |
| `N` | Close the current polygon and name it — needs at least 3 points |
| `Z` | Toggle the existing-zones overlay |
| `S` | Print paste-ready YAML |
| `Q` | Quit |

`S` prints the camera resolution as a comment plus a `zones:` block to paste
under the camera entry in `cameras.yaml`.

Two things to keep in mind while drawing, both from
[ARCHITECTURE](ARCHITECTURE.md#zones-and-the-anchor-point):

**Trace the floor, not the person.** A person's zone is decided at their feet.
Polygons drawn around standing bodies read wrong from any oblique angle.

**Leave no seams.** Adjacent polygons that do not quite meet leave a gap where
detections are tagged `unzoned` — the zone widgets go empty while everything
else looks healthy. The `Z` overlay is how you check.

Order matters too: zones are matched in config order, first match wins, so a
frame-wide zone must be last.

### `inference` mode

Runs the real configured model on the live stream with an overlay — boxes,
class names, confidences, zones, and measured inference FPS. It uses the actual
`ModelRunner`, including the tracker when `use_tracker: true`, so what you see
is what the pipeline sees.

```bash
python3 tools/debug.py --mode inference --camera cam-01 --config config/
```

| Key | Action |
|---|---|
| `Z` | Toggle zone overlay |
| `D` | Toggle detection overlay |
| `Q` | Quit |

Startup logs the model, the camera's active classes, and its zone names — check
those three lines before reading the overlay. This mode answers "is the model
seeing what I think it is" and "are my zones where I think they are". It does
**not** exercise rules, buffering, or ingest, so a healthy overlay with an empty
database points at `rules.yaml`, not at the model.

The FPS here is single-camera and single-process. It is an upper bound, not the
production figure — four cameras share one GPU.

---

## Running as a service

```bash
sudo bash scripts/service.sh install
sudo bash scripts/service.sh start
bash scripts/service.sh logs
```

| Command | Does |
|---|---|
| `install` | Write the systemd unit and enable start-on-boot |
| `start` / `stop` / `restart` | Control the running service |
| `status` | Current state |
| `logs` | Follow the journal |

Install it only once the device runs clean in the foreground. Debugging a
config error through `journalctl` is strictly worse than seeing it on a
terminal.

Restart after **any** config change — nothing is re-read at runtime.

---

## Installation

```bash
bash scripts/install.sh
```

Creates the virtualenv, installs dependencies appropriately for the device
type, and copies all eight config templates from `config/config_sample/` into
`config/`.

**It copies templates only — it never generates config content.** Filling in
real values is a deliberate, reviewable act, not something a script guesses.

On a Jetson the installer uses the JetPack torch already on the system rather
than installing its own. Verify that before going further:

```bash
source .venv/bin/activate
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`False` here means something replaced the vendor torch build — see
[DEPLOYMENT § Traps](DEPLOYMENT.md#1-boxmot-export-destroys-a-jetson-environment).

---

## Reading the logs

Log level comes from `device.yaml` → `log_level`. `INFO` is right for
production; `DEBUG` is per-frame and will bury you.

### At startup

```
VisionEngine Edge — device=node-01  environment=production
detector 'general_coco': loading ./models/yolo26n.engine on cuda
tracker 'general_coco' ready — boxmot BotSort (with_reid=True, use_cmc=False)
tracker 'general_coco': ReID ready — tensorrt backend on cuda (half=True, weights=…engine)
camera 'cam-01': stream ready  960x480
starting 4 camera pipeline(s)
```

Check three things here: the detector loaded the format you expect on the device
you expect; ReID says `tensorrt` and not `pytorch`; and the stream resolution
matches what your zones were drawn against.

Because `botsort_tracker.yaml` is not strictly validated and `BotSort` absorbs
unknown keys, **the startup `BotSort: …` line is the only proof a tracker value
arrived**. A typo is swallowed silently by the file.

### While running

Every 10 seconds, per camera:

```
camera 'cam-01': 10.4 fps (target 15) | 104 frames, 178 detections in 10s
```

This is the number that matters. `fps_target` is a ceiling; this is reality —
and it is the value `frame_rate` in `botsort_tracker.yaml` must be set to.

| Symptom in the log | Usually means |
|---|---|
| fps well under target on every camera | Compute-bound. See [ARCHITECTURE § Benchmark](ARCHITECTURE.md#benchmark) |
| fps fine, detections `0` | Model or class filter, not throughput |
| one camera far below the rest | That stream, not the device |
| `failed to open source` on some cameras | Often an NVR concurrent-stream limit |
| fps degrading over hours | Look at the database — usually missing indexes |

### Health beyond the log

The `nodes` table is the durable view: `cameras_error`, `buffer_pending`, and
`uptime_seconds` per heartbeat. A **rising `buffer_pending`** is the earliest
warning that the backend link is degrading, well before anything else shows it.

`device.yaml` → `health_file` writes the same picture to local JSON on an
interval — for a watchdog that must work while the network is down.

Remember that a device which is *off* writes nothing at all: absence of recent
rows is the alarm, not a row saying "down". The query is in
[DATA_MODEL](DATA_MODEL.md#nodes).

---

## Troubleshooting map

Which tool answers which question:

| Question | Use |
|---|---|
| Is the camera reachable? What resolution? | `--mode view` |
| Where should zone polygons go? | `--mode zones` |
| Is the model detecting correctly? | `--mode inference` |
| Is the config valid? | The one-liner above |
| Is the pipeline keeping up? | The 10-second throughput line |
| Is data reaching the backend? | `buffer_pending` in `nodes` |
| Data is missing but everything looks fine | `rules.yaml` — it gates storage |
| Zone widgets are empty | `unzoned` share — see [DATA_MODEL](DATA_MODEL.md#zone-always-has-a-value) |

---

<div align="center">
<br/>

**[Docs index](README.md)** · **[Configuration](CONFIGURATION.md)** · **[Deployment](DEPLOYMENT.md)**

<br/>
</div>
