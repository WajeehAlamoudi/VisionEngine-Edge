<div align="center">

```
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗  
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝  
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
```

### **Stream Recovery**

<br/>

[![Transport](https://img.shields.io/badge/Transport-RTP%20over%20TCP-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#3-rtp-over-tcp)
[![Recovery](https://img.shields.io/badge/Recovery-reconnect%2C%202s→60s-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#1-reconnect-instead-of-re-reading)
[![Incident](https://img.shields.io/badge/Incident-28%20Sep%202026-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#what-happened)

<br/>

> *What a camera losing its stream does to the pipeline, why it used to be
> permanent, and what recovers it now.*

</div>

---

## What happened

Orbit Commercial Store, 28 September 2026. Four cameras, all reached over RTSP
through one NVR. Three of them stopped producing frames and never came back;
the agent kept running and kept reporting for fifteen hours.

| Camera | Stopped (UTC) | Rows written after |
|---|---|---|
| cam-15 Shop entrance | 15:21:19 | none |
| cam-07 Back door | 15:22:18 | none |
| cam-03 Cashier counter | 19:52:53 | none |
| cam-13 Intersection | — | kept running |

Two failed 59 seconds apart, the third four and a half hours later, and one was
never affected. Each failed exactly once — there was no recovery to lose.

The log said the same thing every 60 seconds for the rest of the night:

```
camera 'cam-15': no output from the DeepStream pipeline — retrying in 60s
```

---

## How it reached the pipeline

The camera loop in [`core/pipeline/pipeline.py`](../core/pipeline/pipeline.py)
reads frames through a runtime and treats a failed read as something to wait
out. That is correct for a frame the source could not produce. It is wrong when
the source itself is gone, and the loop had no way to tell the difference:

```
   read() ──► SourceUnavailable ──► sleep 2s, 4s … 60s ──► read() again
                                                            │
                                    the same dead source ◄──┘
```

Nothing in that circle touches the network. A source whose connection has died
returns the same failure forever, and the camera stays dark until the process
restarts. What made it invisible is that everything else kept working: the
pipeline task was alive, the heartbeat was flushing every 30 seconds, and the
device reported `degraded` rather than `down`.

On the DeepStream runtime it was worse than a dead connection. The appsink was
simply never fed:

```python
sample = self._appsink.emit("try-pull-sample", _PULL_TIMEOUT_NS)
if sample is None:
    self._check_bus()
    raise SourceUnavailable("no output from the DeepStream pipeline")
```

No end-of-stream, no bus error, nothing on the GStreamer bus at all — so
`_failed` was never set and the pipeline believed it was healthy. It was
starving, not broken, which is the hardest failure to notice.

---

## What the evidence showed

Four things, each ruling something out.

**The log.** One `retrying in 2s` per camera, then the backoff climbing to 60s
and staying there. A single failure each, never recovered. Retry pacing comes
from `_retry_delay`, so the first line of each series is the moment the stream
actually stopped.

**The kernel log.** Nothing at either failure minute — no link change, no
interface reset. The device's networking was untouched.

**The ingest worker.** `flushed 1 row(s) → 'nodes'` every 30 seconds without a
gap, all night. The uplink to the backend was fine throughout, so this was never
a general network outage.

**The sockets.** All four RTSP control connections were still ESTABLISHED to the
NVR, and carrying almost nothing:

```
$ sudo ss -tni state established '( dport = :554 )'
... bytes_received:18312  lastrcv:57196496     # 15.9 hours since any data
... bytes_received:88268  lastrcv:3424         # keepalives only, ~140 B/40 s
```

Video at 12 fps is megabytes per minute. Kilobytes over hours means the media
was never on these connections — it was arriving as a separate UDP flow, and
that flow had stopped while the control channel stayed up.

---

## Root cause

`nvurisrcbin` negotiates transport with `select-rtp-protocol`, whose default
(`0`, "rtp-multi") prefers UDP. So the session split in two: control on TCP 554,
media on its own UDP flow.

At the store that flow crossed Wi-Fi and a consumer router doing NAT between two
subnets. A NAT entry for an idle-ish UDP flow ages out on a timer, and when it
does the packets stop arriving with nothing sent to either end — no FIN, no
reset, no error. The TCP control channel survived because TCP mappings are held
far longer and our keepalives touched it every 40 seconds.

That explains every observation: no error anywhere, cameras failing one at a
time rather than together, one camera unaffected, and the device otherwise
perfectly healthy.

---

## The fixes

### 1. Reconnect instead of re-reading

`CameraRuntime.reconnect()` in [`core/model/detector/base.py`](../core/model/detector/base.py)
closes the source and opens it again. The OpenCV runtime inherits it unchanged.
The DeepStream runtime overrides it to clear the state that outlives the
pipeline first — `_failed`, `_link_error`, and `_linked`, which `open()` waits on
to know the source connected and which would otherwise let a stream that never
linked be reported as ready.

The loop still owns the pacing. After each backoff it reconnects rather than
re-reads, and a failed attempt stays in the backoff instead of returning to
`read()` — with no source open, `read()` would raise about a missing pipeline
and hide the camera that is actually missing.

```
   read() ──► SourceUnavailable ──► sleep ──► reconnect() ──► read()
                                                │
                                  still down ───┘  wait longer, try again
```

### 2. A camera that is down at startup is waited for

An `open()` that fails at startup used to abandon that camera for the rest of
the run, so a restart during a camera reboot cost a camera until the next
restart. It now joins the same reconnect loop.

Losing every camera therefore no longer ends the process. That is deliberate: a
device that stays up keeps reporting *which* camera is down, where one that
exits reports nothing and can reach the service's restart limit.

### 3. RTP over TCP

The DeepStream source now asks for `select-rtp-protocol = 4` (TCP only), which
is what the OpenCV runtime has always requested through its ffmpeg options. The
media is interleaved on the RTSP connection, so there is no second flow for a
NAT to forget, and every video packet refreshes the one mapping that matters.

It also changes what a failure looks like. A broken TCP connection is reported;
a vanished UDP flow is not. The recovery in §1 fires on an error rather than on
a five-second starvation timeout.

### 4. A teardown that cannot finish is abandoned

The three fixes above were not enough, and the reason was one line. Rebooting
the NVR with all four cameras live produced `reconnecting in 2s` and then
nothing — no backoff, no further attempt, for as long as the process ran.

A stack dump named it exactly:

```
close (core/model/detector/deepstream/runtime.py:678)
reconnect (core/model/detector/base.py:134)
reconnect (core/model/detector/deepstream/runtime.py:674)
```

All four camera threads sat in `close()`, inside
`pipeline.set_state(Gst.State.NULL)`, which never returned. Reaching NULL joins
the streaming threads, and on this hardware one of them stays in the decoder's
V4L2 teardown when the RTSP socket is dead. Every socket timeout `rtspsrc` has
is bounded — 20s at worst — so this is not a slow wait, it is an unbounded one.
Going to TCP (§3) made it reachable: the media now shares the connection being
torn down, so the teardown waits on the same dead socket.

Nothing after that line runs. The reconnect, the backoff and their log lines are
all correct and all unreachable, which is why the failure looked like missing
retry logic. `request_stop()` calls the same blocking transition from the event
loop, so a camera that cannot be torn down also stalls shutdown.

`close()` now takes the pipeline off the runtime before touching it, flushes the
bus, steps `PAUSED → READY → NULL` rather than jumping to NULL, holds a
process-wide lock so one pipeline unwinds at a time, and does all of it on a
thread it waits on for ten seconds. Past that it gives up, logs, and returns —
`open()` builds a new pipeline and the old one is left to its stuck thread.

The probe is bounded for the same reason. It runs on every `open()`, so an
unbounded `cv2.VideoCapture` against a camera that has just stopped answering
would only move the hang from `close()` to the line after it.

Abandoning a pipeline leaks its socket and decoder context until the process
restarts. One per camera per drop — four for an NVR reboot, nothing after that,
since a later attempt has no pipeline left to release. That is the cost of a
camera that recovers unattended.

---

## What you see now

| Log line | Meaning |
|---|---|
| `... — reconnecting in 2s` | The stream just dropped. First failure |
| `reconnect failed: ... — retrying in 4s` | Camera still unreachable; backoff growing to a 60s ceiling |
| `pipeline did not stop within 10s — abandoning it` | The teardown hung; that pipeline is leaked and a new one is being built. Recovery continues |
| `camera 'X': reconnected` | New connection open, frames flowing, failure count reset |
| `camera 'X': stream ready  960x576` | Resolution re-read after the reconnect |
| `failed to open source: ... — reconnecting in 2s` | Camera was down when the agent started; it is being waited for, not abandoned |

A camera in the reconnect loop has `last_error` set, so the heartbeat counts it
under `cameras_error` and the dashboard shows the device as `degraded`. Once it
reconnects, `last_error` is cleared and it returns to `cameras_active` — without
that, a recovered camera would report the failure it had already survived until
its next detection.

---

## Verifying on a device

Transport — the media should no longer be on UDP at all:

```bash
sudo ss -unap | grep python                 # expect: nothing
sudo ss -tni state established '( dport = :554 )' | grep -oE "bytes_received:[0-9]+"
```

Run the second one twice a few seconds apart. The counters climb by megabytes
when the media is interleaved, and by a few hundred bytes when it is not.

Recovery — drop a camera deliberately and watch it come back:

```bash
journalctl -u visionengine-edge --utc -f | grep -iE "reconnect|stream ready"
```

Unplug the camera or disable its channel on the NVR for 30 seconds, then restore
it. Expect `reconnecting in 2s`, possibly a failed attempt or two, then
`reconnected` and `stream ready`.

Timestamps — `journalctl` parses `--since` in local time and prints in UTC when
given `--utc`, while the agent's own log lines are local and every database
timestamp is UTC. Comparing a log to a row means converting one of them.

---

## The network path

The code survives a bad path; it does not make one good. In order of how much
they remove:

| Change | Removes |
|---|---|
| Edge device wired to the camera network, same subnet as the NVR | Wi-Fi and NAT entirely |
| Intermediate router in AP/bridge mode (LAN IP on the camera subnet, DHCP off, uplink in a LAN port) | The NAT |
| Edge device wired to the nearest AP | The wireless hop |
| RTP over TCP (§3) | A NAT's UDP timeout, permanently |

Two settings on the device itself are worth checking before blaming anything
else. Wi-Fi power saving makes an embedded radio nap between beacons, which bulk
video notices and keepalives do not:

```bash
nmcli con mod "<ssid>" wifi.powersave 2 && sudo nmcli con up "<ssid>"
iw dev <iface> get power_save          # expect: off
iw dev <iface> link                    # signal, band, and channel width
```

On 2.4 GHz, a 40 MHz channel overlaps most of the band and collides with every
neighbour; 20 MHz on a quiet channel carries four sub-streams with room to
spare. `iw dev <iface> survey dump` gives real airtime occupancy rather than a
guess from beacon strength.

Give the NVR a static address. Every RTSP URL in `cameras.yaml` names it, and
nothing in the agent can recover from all four URLs pointing somewhere else.

---

## What is still not covered

**A stream that keeps delivering frames that are wrong** — frozen, black, or the
wrong camera. Reads succeed, so nothing here triggers. Only the data shows it.

**A source that can never open** — wrong URL, wrong credentials, a channel
removed from the NVR. The agent retries once a minute forever, which is visible
in the log and in `cameras_error`, but nothing distinguishes a misconfiguration
from a camera that is simply down.

**Which camera is down, from the dashboard.** The heartbeat payload carries
per-camera id, status and `last_error`, but only the counts are stored, so the
dashboard can say "1 of 4 live" and not which one. Reconstructing it means
reading the log or comparing detection gaps.

---

<div align="center">
<br/>

**[Documentation index](README.md)** · **[Architecture](ARCHITECTURE.md)** · **[Tools](TOOLS.md)**

<br/>
</div>
