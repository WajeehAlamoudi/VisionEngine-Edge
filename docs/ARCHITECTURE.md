<div align="center">

```
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗  
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝  
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
```

### **Architecture & Inference**

<br/>

[![Detect](https://img.shields.io/badge/Detect-YOLO-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](https://ultralytics.com)
[![Track](https://img.shields.io/badge/Track-BoT--SORT%20%2B%20OSNet-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](https://github.com/mikel-brostrom/boxmot)
[![Accel](https://img.shields.io/badge/Accel-TensorRT%20%7C%20DeepStream%20%7C%20CoreML-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#detector-backends)

<br/>

> *What happens between a frame arriving and a row landing — and why each choice was made.*

</div>

---

## The pipeline

One `CameraPipeline` per enabled camera, each an independent asyncio task. A
camera failing does not stop the others.

```
  ┌── capture ─────────  RTSP / USB / file, timestamped at read
  │
  ├── detect ──────────  Detector backend → boxes, classes, confidences
  │                      filtered to the camera's declared classes
  │
  ├── track ───────────  BoT-SORT: motion prediction + appearance match
  │                      → assigns a stable track_id           [optional]
  │
  ├── enrich ──────────  anchor point → zone name, normalized coords,
  │                      camera + frame context, capture timestamp
  │
  ├── evaluate ────────  RulesEngine: does any enabled rule match?
  │                        no  → discard
  │                        yes → row queued for the detections table
  │                              alert if notify + cooldown cleared
  │
  ├── buffer ──────────  local SQLite spool, survives network loss
  │
  └── ingest ──────────  batched POST to the backend
             └── notify   live webhook + log, separate path, not persisted
```

Two properties worth internalising:

**Timestamping happens at capture**, so rows replayed hours after an outage
still carry the moment they happened. Nothing downstream infers time from
arrival order.

**Rules gate storage, not just alerts.** A detection matching no enabled rule is
discarded before the buffer. This is the difference between a device that writes
a million useful rows a day and one that writes ten million, most of them noise.

---

## Detector backends

`models.yaml` → `device` selects the backend. There is no autodetection beyond
`auto` resolving among torch devices.

| `device` | Backend | Weight format | Notes |
|---|---|---|---|
| `auto` | Ultralytics | `.pt` / `.engine` | Resolves to cuda → mps → cpu, in that order |
| `cpu` | Ultralytics | `.pt` | Baseline. Works everywhere |
| `cuda` | Ultralytics | `.pt` / `.engine` | Jetson and desktop NVIDIA |
| `mps` | Ultralytics | `.pt` | Apple Metal |
| `coreml` | Ultralytics | `.mlpackage` | Execution target is fixed **at export time**, not by `device=` |
| `deepstream` | DeepStream | `.engine` / `.onnx` | Same NVIDIA hardware as `cuda`, different runtime on it. **Tracks too** when `use_tracker` is on — see below |

Each backend imports its SDK **lazily, inside its own `load()`/`infer()`** —
never at module top level. That is what lets a machine with no DeepStream
install run every other backend normally, and vice versa. Adding a backend means one
entry in `core/model/detector/registry.py`; nothing above it changes.

### Detectors that track

`Detector` is normally stateless and detection-only: `infer()` returns
`track_id=None`, and the `tracker/` layer adds ids afterwards. That split is
what lets any tracker sit behind any detector.

DeepStream cannot be split that way. `nvinfer` (detection) and `nvtracker`
(tracking) are elements in one GStreamer pipeline, operating on the same GPU
buffers in the same pass; `nvtracker` tracks the boxes `nvinfer` just produced
and cannot accept detections from anywhere else. So it lives *inside*
`DeepStreamDetector` rather than behind the `Tracker` ABC.

Two flags carry that upward, and they answer genuinely different questions:

```python
class Detector(ABC):
    tracks_internally: bool = False   # assigns track_id itself
    shareable: bool = True            # one instance can serve several cameras
```

- **`tracks_internally`** is read by `ModelRunner.load()`, which then skips
  building a `Tracker`. Building one would re-track already-tracked boxes and
  throw the pipeline's ids away. It is read through
  `registry.tracks_internally(cfg)` rather than off the class, because it also
  depends on config: DeepStream with `use_tracker: false` builds no `nvtracker`
  at all and is then an ordinary detection-only backend.
- **`shareable`** is read by `ModelRegistry.load_for_cameras()`, which then
  gives the model a dedicated instance per camera. Tracking state is one reason
  to be unshareable, but not the only one — a DeepStream pipeline negotiates
  caps once for a single frame size, so a second camera at another resolution
  could not use it whether or not tracking is on.

Both are answered from class attributes without constructing anything, so
neither decision costs an SDK load.

Both backends route their raw integer ids through the same `StableIdMap`
(`core/model/stable_id.py`), so every `track_id` reaching the database is a
UUID regardless of which produced it.

### Model formats

The rule that catches people out: **where an artifact is built determines where
it runs.**

| Format | Built | Portable? |
|---|---|---|
| `.pt` | anywhere | Yes — the fallback that always works |
| `.engine` (TensorRT) | **on the target device** | No. Tied to that GPU and TensorRT version |
| `.mlpackage` (CoreML) | on a Mac | Its compute unit is baked in at export |

Export commands are in
[DEPLOYMENT § Stage 3](DEPLOYMENT.md#32-detector-model).

### Why TensorRT is worth the inconvenience

A `.pt` runs the graph as PyTorch wrote it: one kernel launch per layer, generic
kernels, activations round-tripping to global memory between every op.

TensorRT compiles the graph **for one specific GPU**. It fuses conv-bn-relu into
single kernels, picks tuned implementations by benchmarking candidates on the
actual hardware, plans memory reuse ahead of time, and emits a fixed execution
plan. That planning is why the build takes minutes and why the result is not
portable — the artifact encodes decisions about *that* GPU's SM count, memory
bandwidth, and driver.

### Precision

`half: true` requests FP16 — half the memory traffic, and on hardware with
tensor cores a genuinely faster datapath. INT8 goes further but needs a
calibration dataset, since it must learn each tensor's dynamic range.

**Neither is a reliable win, and this deployment measured both losing.** The
reason is where the time actually goes:

- **Arithmetic-bound** work — big convolutions over large feature maps — gets faster with lower precision. There is real math to reduce.
- **Launch-bound** work — many small layers — does not. The GPU spends its time in per-kernel overhead, and halving the arithmetic inside each kernel changes nothing.

A small network is usually launch-bound. That is exactly the case where FP16 and
INT8 disappoint while **layer fusion** — which removes launches rather than
shrinking math — wins enormously. Measure the stage before choosing the
optimisation; see [the benchmark](#benchmark).

For an `.engine`, precision is fixed when the engine is built. `half:` in
`models.yaml` has no effect on it.

---

## Tracking

Detection is per-frame and stateless: it says *a person is here*, never *this is
the same person as last frame*. Tracking supplies identity, and identity is what
turns detections into analytics — unique visitors, dwell time, zone-to-zone
movement all require it.

### The stack

Three separate things, often confused:

| | What it is | Role here |
|---|---|---|
| **YOLO** | Detection model | Finds objects in a frame |
| **boxmot** | Tracking *library* | Hosts several tracker implementations |
| **BoT-SORT** | Tracking *algorithm*, one of boxmot's | The one we use |
| **OSNet** | ReID network | Produces the appearance embedding BoT-SORT matches on |

So: YOLO detects, BoT-SORT (from boxmot) associates across frames, OSNet tells
BoT-SORT what each person *looks like*. Switching algorithm means changing which
boxmot tracker is constructed — not swapping the detector.

### How BoT-SORT associates

Each existing track has a Kalman filter predicting where it should be in the
next frame. New detections are matched against those predictions on two signals:

1. **Motion** — IoU between the detection box and the prediction.
2. **Appearance** — cosine distance between the detection's OSNet embedding and the track's.

Motion alone breaks whenever a person moves far between frames — which is
exactly what a low frame rate causes — or when two people cross. Appearance
survives both, which is why `with_reid: true` matters far more on an edge device
at 10 fps than in a 30 fps benchmark.

The association runs in passes: high-confidence detections first, then a second
pass over the low-confidence ones that rescues partially occluded boxes the
first pass missed.

### Track memory, and the one number people get wrong

A track that stops matching is not deleted immediately — it is remembered so the
person can reclaim their id after walking behind a pillar:

```
buffer_size = int(frame_rate / 30 × track_buffer)
memory in seconds = track_buffer / 30      ← only while frame_rate is accurate
```

`frame_rate` in `botsort_tracker.yaml` must be the **measured** rate, not
`fps_target`. It does nothing but scale this window, and a stale value silently
changes it in whichever direction is worse:

- **Too low** — set to 4 while running at 12 — the buffer shrinks and people lose their id on any brief occlusion. Unique-visitor counts inflate.
- **Too high** — a long-abandoned track stays revivable and can be handed to a different person. Counts deflate and journeys get stitched wrongly.

Measure first, configure second. This is why
[DEPLOYMENT](DEPLOYMENT.md#35-run-then-measure-then-tune) puts `frame_rate`
after the first run rather than in the initial config pass.

### `track_id` is a UUID, not boxmot's integer

boxmot numbers tracks from 1 and resets on every process start. Persisting that
integer would let a track from today collide with an unrelated one after a
restart next week. Each raw integer is mapped to a fresh UUID the first time it
is seen, and the map starts empty on every run — so ids from a previous run can
never resurface. The cost is that a restart splits one visit into two ids; the
alternative was silent cross-day collisions.

### Which ReID backends are selectable

Only **`pytorch`** and **`tensorrt`**. boxmot ships six.

The other four (`onnx`, `openvino`, `tflite`, `torchscript`) each declare a pip
requirement that boxmot **auto-installs on first load** when unsatisfied. On a
device whose CUDA stack comes from the OS rather than pip — a Jetson running
JetPack — that installer pulls a generic torch wheel whose bundled CUDA runtime
the driver cannot use, and the GPU goes dark. Selecting one raises at startup
instead, with a message saying why.

The `tensorrt` path disables that installer explicitly: it looks for a pip
package named `nvidia-tensorrt`, which does not exist for Jetson because
TensorRT ships with the OS. TensorRT is trusted as a system library, exactly as
the detector backends trust it. If it is genuinely missing, the `ImportError`
is the correct failure.

`reid_device` is a **torch** device and is not necessarily the detector's. A
CoreML or DeepStream detector still needs its ReID on cpu or cuda — those are not
torch devices, and reusing them used to raise at model load.

### Dynamic batch is mandatory

One crop is embedded per detection, so three people in frame is a batch of
three. An engine exported without a dynamic batch dimension accepts exactly one
and works perfectly in single-person testing, then fails the moment a second
person appears. Export with a shape profile —
[DEPLOYMENT § 3.3](DEPLOYMENT.md#33-reid-model).

---

## Zones and the anchor point

Zone membership is tested at **one point**, not against the whole box.

```
   ┌───────────┐
   │           │
   │  person   │        person  → anchor = bottom-centre  (feet)
   │           │        others  → anchor = box centre
   └─────●─────┘
      anchor
```

For `person` the anchor is `((x1+x2)/2, y2)`; every other class uses the
geometric centre.

The reason is physical. A standing person's box runs from feet to head, so its
centre sits around chest height — a point in mid-air, metres from where they are
standing in a floor plan. Test there and someone at the near edge of a zone
reads as outside it while someone in the background reads as inside, with the
error growing as the camera gets more oblique. Feet are where the person
actually is. A car or a bag has no such asymmetry, so the centre is right for
them.

**The practical consequence: draw zone polygons on the floor**, tracing walkable
area, not around a standing body's height.

Assignment is a ray-casting point-in-polygon test, **in config order, first
match wins**. So a frame-wide zone listed first swallows every zone after it,
and a detection outside all polygons is tagged `unzoned` — a real value, not a
null. `zones: "*"` skips subdivision entirely and tags everything `full_frame`.

Adjacent polygons that do not quite meet leave a seam, and everything landing in
it becomes `unzoned` while tracking works perfectly — the zone widgets read
empty and nothing looks broken. Check the `unzoned` share before trusting zone
analytics ([DATA_MODEL](DATA_MODEL.md#zone-always-has-a-value)).

---

## Delivery

### The buffer

Rows go to a local SQLite spool before the network. When the backend is
unreachable they accumulate and replay later, still carrying capture
timestamps. `max_size_mb` is the point at which the **oldest rows are dropped**
and `delete_after_hours` expires them regardless — both are data-loss policy,
sized for how long an outage you intend to survive.

`buffer_pending` in the `nodes` table is the live view of this. A rising value
is the earliest signal that the link is degrading.

### Two paths, deliberately different

| | Ingest | Webhook |
|---|---|---|
| Transport | Batched POST | Immediate POST |
| Durability | Buffered, retried | Neither |
| Loses | Nothing, up to buffer limits | The live alert only |
| Feeds | `detections`, `notifications`, `nodes` | Frontend WebSocket |

They are separate because they are answering different questions. Ingest is the
record and must not lose rows; the webhook is a doorbell and a late one is worse
than none. A webhook outage loses live alerts, never stored rows — the
`notifications` row still lands through the buffered path.

---

## Benchmark

Jetson Orin NX · JetPack 6.2 · TensorRT 10.7 · 4 cameras @ 960×480 · BoT-SORT
with appearance matching on throughout · measured over 10-second windows with
people in frame · **2026-08-18**.

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

Three things this established.

**The ReID model, not the detector, was the bottleneck.** OSNet x0.25 has ~200k
parameters — a rounding error next to the detector — and took twice as long. It
was launch-bound, not arithmetic-bound: many tiny layers, each costing more in
launch overhead than in math.

**Which is why FP16 on it changed nothing and fusion changed everything.**
Halving arithmetic that was not the constraint does nothing; removing the
launches gave 32×. INT8 on the detector was tried first and also gained nothing.

**The bottleneck has moved.** ReID fell from 65% of the frame to 6%; the
detector rose from 34% to 90%. Detector-side work that was worthless before —
INT8, a smaller `input_size` — is now where the remaining headroom is.

The method generalises better than the numbers: two guesses failed, then
per-stage timing found the answer in one pass. Profile the stages before
choosing an optimisation.

---

<div align="center">
<br/>

**[Docs index](README.md)** · **[Configuration](CONFIGURATION.md)** · **[Data model](DATA_MODEL.md)** · **[Deployment](DEPLOYMENT.md)**

<br/>
</div>
