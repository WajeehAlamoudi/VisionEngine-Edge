<div align="center">

```
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
```

### **Tracker History & FPS Degradation**

<br/>

[![Scope](https://img.shields.io/badge/Scope-BoxMOT%2019%20ByteTrack-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#tracker-specific-scope)
[![Validation](https://img.shields.io/badge/Validation-Local%20synthetic%20load-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#validation-results)
[![Measured](https://img.shields.io/badge/Measured-12%20Oct%202026-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#validation-results)

<br/>

> *Why expired tracking history can slow every frame, how the ByteTrack-specific
> cleanup bounds that work, and what the tests do—and do not—prove.*

</div>

---

## When to use this guide

Investigate this issue when FPS degrades with process uptime, even in quiet
scenes. The observed deployment used four cameras, YOLO26s TensorRT, and
BoxMOT 19.0.0 ByteTrack without ReID on a Jetson Orin NX 8 GB.

During the October 12 investigation, all streams were below their 12-FPS
target. Clocks were already at maximum, sampled GPU utilization was low,
and video was queued unread. Approximately 91% of samples in a 15-second
Python-lock-held profile were in the removed-history lookup construction.
This percentage describes that filtered profile, not total CPU or GPU time.
H264 decoding errors were also present; this fix does not establish their cause.

---

## Root cause

ByteTrack maintains active, temporarily lost, and removed track collections.
Lost tracks can be recovered within the configured absence window. Removed
tracks have expired and are not a long-term person-recognition database.

The installed ByteTrack update performs this housekeeping:

```python
self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
self.removed_stracks.extend(removed_stracks)
```

`sub_stracks` builds a new set of IDs from the entire removed list, then
filters the lost list against it. Its cost is O(R + L) per update, where R is
historical removed entries and L is lost entries. The historical list grows
without a bound in the inspected implementation—even when L is zero.

This is not appearance matching or new-ID allocation. With 100,000 removed
entries and three lost tracks, the code reads 100,000 entries before doing
three membership checks. It repeats that work on the next processed frame.

Marking a track removed changes its state; it does not automatically remove
its reference from `lost_stracks`. Newly removed entries are appended after
the subtraction, so some are needed for the following update's cleanup.
Clearing the entire removed list indiscriminately would bypass that cleanup.

---

## Minimal solution

The application-level change lives in
[`BoxMotTracker`](../core/model/tracker/boxmot_tracker.py), not in the installed
dependency. After each successful ByteTrack update, `_prune_removed_history`:

1. Collects IDs still present in the lost pool.
2. Retains only removed entries whose IDs are still in that pool.
3. Discards obsolete removed-history references.

Newly expired entries survive long enough for the next cleanup. Once removed
from the lost pool, they no longer need to be retained as historical objects.
When `per_class` is enabled, cleanup operates on each class's saved pools and
then refreshes BoxMOT's aggregate collections.

The first update against an already huge list still pays the historical scan
and pruning cost. Subsequent updates no longer scan lifetime history. This
bounds retention by pending cleanup, not by a fixed number of identities;
busy scenes still require more matching work than empty ones.

No changes are made to matching thresholds, absence buffers, frame-rate
settings, active tracks, UUID generation, database rows, or rule filtering.

---

## Tracker-specific scope

| Component | Treatment |
|---|---|
| BoxMOT 19 ByteTrack | Cleanup enabled only when the resolved algorithm is `bytetrack` |
| ByteTrack with `per_class: true` | Each class-local lost/removed pool is cleaned separately |
| Other BoxMOT algorithms | Unchanged; do not apply ByteTrack assumptions to their internal state |
| DeepStream / NvTracker | Unchanged; this code path is not used |
| Shared `StableIdMap` implementation | Unchanged; separate concern described below |

A unified tracker configuration does not imply identical lifecycle behavior.
Before extending this cleanup to another algorithm—or upgrading BoxMOT—inspect
its removal order, recovery logic, ID accessor, and per-class storage, then
test those behaviors. Another tracker may intentionally retain history for
appearance recovery or other purposes.

### Stable UUID mapping is separate

[`stable_id.py`](../core/model/stable_id.py) is shared code used by BoxMOT and
DeepStream, with separate map instances per camera/tracker. It maps raw integer
IDs to UUIDs so integer reuse after a restart does not normally collide with
persisted IDs. Its dictionary lookup does not scan all previous entries.

The mapping currently retains entries for its lifetime, so it is a separate
source of memory growth. This change does not prune it. Any future pruning
must preserve mappings for tracks that can still recover; absence from one
frame's output is not sufficient evidence that an ID can be discarded.

---

## Validation results

Measured October 12, 2026 on a local Windows 11 workstation, Python 3.12.6,
BoxMOT 19.0.0. CPU model was not recorded. These are comparative tracker-only
measurements, not a portable hardware throughput specification.

- Four concurrent camera threads, scheduled at 12 FPS each, 30 seconds per version.
- 960 × 576 synthetic frames; no detector, RTSP decoding, GPU inference, or ingest.
- Deployment tracker parameters: `track_buffer=75`, `frame_rate=12`,
  `min_conf=0.2`, `track_thresh=0.5`, `match_thresh=0.8`, `per_class=false`.
- 100,000 distinct removed `STrack` objects/IDs seeded per camera (400,000 total).
  Shallow copies share unused history arrays: this tests scan cost, not realistic RAM usage.
- Moving detections, low-confidence association, short occlusions, and turnover.
  Each 120-slot cycle has detections for 72 slots and absence for 48 slots;
  the counts below are maxima, not continuous occupancy.
- Overdue input slots are skipped rather than queued indefinitely. Reported FPS
  is completed updates divided by the 30-second run, including initial cleanup.

| Maximum people per camera | Before: FPS | After: FPS | Before: p95 update ms | After: p95 update ms |
|---|---:|---:|---:|---:|
| 0 | 4.67 | 12.00 | 380.98 | 2.04 |
| 4 | 2.87 | 11.83 | 537.63 | 21.98 |
| 12 | 2.23 | 11.83 | 880.58 | 41.20 |
| 24 | 1.77 | 11.83 | 1296.05 | 54.60 |

All processed updates in the unmodified run missed their scheduled 83.3 ms
deadline. The fixed run had 2–3 deadline misses per camera; its first updates
took 134–578 ms while disposing of the seeded backlog. Final removed-history
counts were zero with cleanup versus over 100,000 per camera without it.

Separate correctness checks passed:

- 2,000 synthetic frames per mode (normal and per-class): identical tracking
  outputs; removed history peaked at two entries versus 800 final entries
  without cleanup. Short-gap UUID recovery and post-expiration new IDs passed.
- 100,000 initial removed IDs, 120 comparison frames per mode: identical
  outputs across disappearance, expiration, and re-entry.
- Other-algorithm bypass, syntax, and seven existing configuration tests passed.

Tests were run outside the repository, as requested; the external harness and
raw results are not bundled with this guide. These short synthetic tests do
not constitute a seven-day soak test, real-video accuracy evaluation, or proof
that every source of memory growth has been removed.

---

## Deployment verification

At the time of this report, the fix is locally validated, not production-verified.
Deploy through the normal reviewed Git workflow; do not edit installed BoxMOT
files or a live Jetson checkout as an ad hoc patch. A service restart interrupts
processing and starts new tracking identities, so schedule it explicitly.

After approved deployment, compare per-camera FPS, process RSS, decoding/error
logs, and track continuity during both quiet and busy periods. Recheck after
24–48 hours and longer operation. A quiet-scene improvement alone is not a
complete validation. Investigate decoding, inference, and other bottlenecks
separately if FPS still drops; changing the absence buffer is not this fix.

---

## Related upstream work

- [Supervision issue #1164](https://github.com/roboflow/supervision/issues/1164)
  reports the same unbounded-history cost and proposes retaining only the
  latest update's removed tracks.
- [Ultralytics cleanup implementation](https://docs.ultralytics.com/reference/trackers/utils/stracks/)
  documents a bounded removed-history buffer. This is a different implementation,
  not evidence that the installed BoxMOT version already contains a fix.
- [Original ByteTrack PR #249](https://github.com/FoundationVision/ByteTrack/pull/249)
  proposes age-based pruning; it was open when consulted, not a deployed fix here.

---

<div align="center">
<br/>

**[Documentation index](README.md)** · **[Configuration](CONFIGURATION.md)** · **[Runtime stability](RUNTIME_STABILITY.md)** · **[Tools](TOOLS.md)**

<br/>
</div>
