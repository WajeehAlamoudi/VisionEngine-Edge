<div align="center">

```
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗  
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝  
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚════╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
```

### **Runtime Stability**

<br/>

[![Subject](https://img.shields.io/badge/Subject-Choosing%20a%20runtime-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#which-runtime-for-which-site)
[![Open](https://img.shields.io/badge/ReID%20crash-under%20investigation-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#the-reid-cuda-crash--open)
[![Measured](https://img.shields.io/badge/Measured-29%20Sep%202026-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#what-was-measured)

<br/>

> *Which runtime a site should use, what each one costs, and an open GPU
> failure that is not yet resolved.*

</div>

---

## Which runtime for which site

Both runtimes were run against the same four cameras on the same device, on
the same day, through the same NVR reboot. They did not behave the same way.

| | deepstream | ultralytics |
|---|---|---|
| Decode | NVDEC, stays in GPU memory | ffmpeg on CPU, copied to host |
| Frame rate under load | steady 11.9–12.0 | 5.0–12.0, varies with the scene |
| Recovery from a dropped stream | needed four fixes; ends in a process restart | worked unchanged |
| Code to make recovery work | four commits | one line |
| Suits | many streams per device | a handful of streams |

**Stream count is the deciding factor, not connection type.** DeepStream reads
RTSP perfectly well; what it buys is density — sixteen to sixty-four streams
batched through one `nvstreammux` with nothing crossing to host memory. At four
cameras that headroom is unused, and what remains is its operational cost: see
[STREAM_RECOVERY](STREAM_RECOVERY.md) for the teardown deadlock that cost a
store fifteen hours of blind cameras.

The reverse is also true. The ultralytics path decodes every stream on the CPU
and embeds one ReID crop per detected person, so its cost rises with how busy
the scene is. Two cameras in the same ten seconds:

```
cam-07: 11.9 fps  |    0 detections
cam-03:  5.0 fps  |  229 detections
```

That is the trade in one line. A steady rate, or a runtime that recovers on its
own. For four cameras, recovery is worth more.

---

## What was measured

Jetson Orin NX **8 GB** (not 16 — the figure that matters most and the one
easiest to assume wrong), JetPack with TensorRT 10.7, 40 W power mode: six of
eight cores online, capped at 1.5 GHz.

| | |
|---|---|
| Model | YOLO26 medium, TensorRT FP16, 640×640 |
| Cameras | 4 × 960×576 RTSP, `fps_target: 12` |
| Frame rate | 11.3–12.0 idle, dipping to 5.0 under detections |
| GPU memory | ~1.5 GB of TensorRT contexts |
| Host memory | 3.3 GB used **before the agent starts** — GNOME desktop |

Two numbers worth keeping in mind on this board. Memory is unified, so GPU
allocations come out of the same 8 GB as everything else. And `tegrastats`
reported `lfb 37x4MB` — the largest free block is 4 MB, which is why `trtexec`
can fail with `LLVM ERROR: out of memory` while `free` shows 4 GB available.

---

## The ReID CUDA crash — open

**Status: not resolved.** Mitigations are deployed; none is confirmed as the
cause. This section is a record of what has been eliminated, so the next person
does not re-test it.

### What happens

With four cameras on the ultralytics runtime and BoT-SORT using a TensorRT ReID
engine, the process dies once people appear:

```
[TRT] [E] IExecutionContext::executeV2: Error Code 1: Cask (Cask convolution execution)
camera 'cam-15': inference error: tracker/ReID update failed:
    CUDA error: an illegal memory access was encountered
```

It corrupts the **process-wide** CUDA context, so all four cameras fail in the
same second — including one with `0 detections` that was running no ReID at
all. Every reconnect after that fails identically, because the context is gone
and only a restart rebuilds it.

Timing is the strongest clue: hours of stability while detections were zero,
then a crash **two seconds** after the fourth camera came up with people in
frame.

### Ruled out, with evidence

| Theory | Evidence against |
|---|---|
| ReID batch exceeded the engine profile | Profile is `min=1 opt=16 max=100`; it was fed 4–5 |
| Total memory exhaustion | 4.0 GB available and 11 GB swap unused at idle |
| Four cameras sharing one execution context | Four separate `ReID ready` lines — one backend each |
| The detector (YOLO) engine | Always batch 1; every failure names `executeV2` in ReID |

### Still open

**A stale output pointer in boxmot's TensorRT backend.** `load_model()` sizes
the output binding at the engine's *opt* batch and records its address once in
`binding_addrs`. `forward()` then calls `resize_()` on that tensor whenever the
batch changes, and refreshes only the **input** address. A resize that grows
past the current storage reallocates — new memory, old pointer still in
`binding_addrs` — and TensorRT writes results into freed device memory.

It fits the parts that nothing else explained: why it survives at first
(shrinking reuses the same storage), why it is intermittent (reading or writing
past a tensor only faults when it lands on an unmapped page), and why more
cameras fail sooner (more batch-size changes per second).

Unproven. The test that settles it is **one camera** with TensorRT ReID: if a
single camera still crashes, concurrency is out and this is what remains.

### Deployed mitigations

Neither is a confirmed fix.

**One engine load per camera.** Four cameras produced eight
`Loading … .engine` lines — `BaseModelBackend.__init__` calls `load_model()`,
then `TensorRTBackend.__init__` calls it again, each building a full engine and
context. On an 8 GB board that is worth reclaiming. Verify after any change to
this path:

```bash
PID=$(systemctl show -p MainPID --value visionengine-edge)
journalctl _PID="$PID" --utc | grep -c "Loading .*\.engine"
```

One per camera. **Zero means the skip removed the only real load** and the
backend is unloaded — revert immediately.

**A process-wide lock around `update()`** while ReID runs on TensorRT. This
costs throughput: ReID for every camera queues behind one lock, which adds to
the frame-rate variance above. If the stale-pointer theory is confirmed, this
lock is not the fix and should come out.

### If it crashes in production

Set `with_reid: false` in the tracker config and restart. Detection, zones and
rules keep working; only appearance matching is lost, so identities break more
often across occlusion. Better than a blind store.

`reid_backend: pytorch` with `osnet_x0_25_msmt17.pt` keeps ReID and avoids the
TensorRT backend entirely — a different and much lighter network, so
embeddings will not match anything recorded previously.

---

## Tuning that is not about the crash

**`frame_rate` in the tracker config must be the measured rate**, not
`fps_target`. boxmot remembers a lost track for `frame_rate / 30 × track_buffer`
frames, so a stale value silently changes the window. Keep `900 / fps` to hold
30 frames: 75 at 12 fps, 90 at 10, 113 at 8.

**`confidence_threshold` does not transfer between models.** NvDCF's
`minDetectorConfidence: 0.1894` is right for PeopleNet, which clusters and
thresholds inside nvinfer first. Applied to a COCO YOLO it accepts every box
over 19 % — 229 detections in ten seconds on a cashier counter, which is shelves
and shadows, and it drags the frame rate down with ReID work on each one.

**Headroom on this board.** 40 W mode leaves two cores offline and clocks the
rest at 1.5 GHz while ffmpeg decodes four streams on CPU; `nvpmodel -m 0` lifts
that. And a desktop session holding 3.3 GB on an 8 GB edge device is the largest
single saving available.

---

<div align="center">
<br/>

**[Documentation index](README.md)** · **[Stream recovery](STREAM_RECOVERY.md)** · **[Configuration](CONFIGURATION.md)**

<br/>
</div>
