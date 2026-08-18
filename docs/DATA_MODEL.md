<div align="center">

```
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗  
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝  
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
```

### **Data Model**

<br/>

[![Store](https://img.shields.io/badge/Postgres-per--branch%20schema-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](https://postgresql.org)
[![Tables](https://img.shields.io/badge/Tables-4-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#tables)

<br/>

> *Every column the edge writes, and what it actually means.*

</div>

---

Each branch owns a Postgres schema named `<branch>_branch_<hex>`. Nothing is
shared between branches — no cross-branch keys, no global tables.

```
<branch>_branch_<hex>
├── detections        one row per detected object per frame     ← the record
├── notifications     one row per fired alert                   ← the event log
├── nodes             one row per device heartbeat              ← the telemetry
└── dashboard_config  one row per dashboard widget              ← the UI
```

The DDL that creates them is in
[DEPLOYMENT § Stage 2](DEPLOYMENT.md#stage-2--provision-the-branch-schema).
This guide is about meaning, not creation.

---

## Table names are configurable

`detections` and `notifications` are **conventions, not fixed names**. The
actual destination comes from config:

- `cameras.yaml` → `raw_table` (and per-rule `routing[].table`) for detections
- `rules.yaml` → `notifications_table` per rule

One camera can write to `detections_lobby` while another writes to
`detections_yard`. Every table named this way needs the same columns and the
same indexes. `nodes` and `dashboard_config` are fixed.

---

## Tables

### `detections`

One row per object, per frame, that matched an enabled rule. The analytical
record — high volume, and the only table you should count people from.

| Column | Type | Meaning |
|---|---|---|
| `id` | `BIGSERIAL` | Surrogate key. Carries no ordering guarantee across devices |
| `camera_id` | `TEXT` | The `id` from `cameras.yaml`. Stable — use for joins and filters |
| `camera_name` | `TEXT` | Human label. Free to change; never key on it |
| `model_id` | `TEXT` | Which `models.yaml` entry produced this. Lets you segment by model version |
| `track_id` | `TEXT` | UUID for one identity on one camera. See [below](#track_id) |
| `class` | `TEXT` | Class name exactly as the model declared it in `models.yaml` |
| `confidence` | `NUMERIC` | 0.0–1.0, rounded to 4 dp |
| `bbox_x1` `bbox_y1` | `INTEGER` | Top-left corner, pixels |
| `bbox_x2` `bbox_y2` | `INTEGER` | Bottom-right corner, pixels |
| `bbox_w` `bbox_h` | `INTEGER` | Width and height, pre-computed so SQL never subtracts |
| `anchor_x` `anchor_y` | `NUMERIC` | The point zone membership was tested at. See [below](#the-anchor-point) |
| `anchor_x_norm` `anchor_y_norm` | `NUMERIC` | Same point as 0.0–1.0 of frame size |
| `frame_w` `frame_h` | `INTEGER` | Frame size this row was measured in |
| `zone` | `TEXT` | Zone name, or `full_frame`, or `unzoned`. Never null |
| `ts` | `TIMESTAMPTZ` | **Capture** time, taken at frame read — not insert time |

`class` is a reserved word in some SQL dialects. Quote it: `"class"`.

### `notifications`

One row per alert that actually fired. Everything `detections` has, plus:

| Column | Type | Meaning |
|---|---|---|
| `rule_name` | `TEXT` | The `rules.yaml` rule that matched |
| `severity` | `TEXT` | `critical` \| `warning` \| `info` — set by the rule, not inferred |
| `message` | `TEXT` | The rule's message template with placeholders resolved |

The remaining columns are copied from the detection that triggered it and mean
exactly the same thing.

### `nodes`

One row per heartbeat, per device. Device telemetry, not vision data — this is
what an operations dashboard reads.

| Column | Type | Meaning |
|---|---|---|
| `device_id` | `TEXT` | From `device.yaml`. Groups a device's history |
| `name` `location` | `TEXT` | From `device.yaml`. Descriptive |
| `status` | `TEXT` | Overall device state at this beat |
| `cameras_active` | `INTEGER` | Cameras streaming successfully |
| `cameras_error` | `INTEGER` | Cameras enabled but failing. **Nonzero is your alarm** |
| `detections_total` | `INTEGER` | **Cumulative since process start**, summed over cameras. Resets to 0 on restart |
| `buffer_pending` | `INTEGER` | Rows waiting to reach the backend. Rising = the link is degraded |
| `uptime_seconds` | `NUMERIC` | Seconds since process start. A drop means it restarted |
| `ts` | `TIMESTAMPTZ` | Heartbeat time |

Two traps here. `detections_total` is **cumulative**, so charting it raw draws a
sawtooth across restarts — take a delta, or just use `detections` for counting.
And a device that is *off* writes nothing at all, so absence of recent rows is
the signal, not a row saying "down":

```sql
SELECT device_id, MAX(ts) AS last_seen, now() - MAX(ts) AS silent_for
FROM   <schema>.nodes
GROUP  BY device_id
HAVING now() - MAX(ts) > INTERVAL '5 minutes';
```

### `dashboard_config`

One row per widget. The dashboard's own state, written by the UI rather than
the edge.

| Column | Meaning |
|---|---|
| `widget_id` | Stable identifier |
| `title` | Displayed heading |
| `widget_type` | Renderer — `stat`, `bar`, `line`, `table`, … |
| `config` | `JSONB`. Holds the `source` block the backend compiles to SQL |
| `refresh_seconds` | Auto-refresh interval. `NULL` = load once, never poll. Minimum 5 |
| `position` / layout | Grid placement |

`refresh_seconds` is a **column**, not a key inside `config`.

---

## The concepts behind the columns

### `track_id`

A UUID assigned the first time the tracker sees an identity. Four properties
decide every query you write against it:

**Scoped to one camera.** The same person walking past two cameras gets two
unrelated ids. Zone-level aggregates work across cameras; individual journeys
do not.

**Regenerated on restart.** The tracker's internal counter resets, and each raw
integer is mapped to a fresh UUID, so a restart splits one visit into two ids.
This is deliberate — reusing the raw integer would let today's track collide
with an unrelated one next week. It means a `COUNT(DISTINCT track_id)` spanning
a restart over-counts slightly.

**Lost to occlusion.** A person hidden longer than `track_buffer / 30` seconds
returns as a new id. Undercounting people usually means that window is too
short, not that detection failed.

**Null when tracking is off.** With `use_tracker: false`, every row has
`track_id = NULL`. Filter with `WHERE track_id IS NOT NULL` in identity queries.

### The anchor point

Zone membership is not tested at the centre of the box.

```
   ┌───────────┐
   │           │
   │  person   │        person  → anchor = bottom-centre  (where the feet are)
   │           │        others  → anchor = box centre
   └─────●─────┘
      anchor
```

For `person` the anchor is `((x1+x2)/2, y2)`; for every other class it is the
geometric centre. The reason is physical: a standing person's box extends from
their feet to their head, and the box centre sits around chest height. Test
there and someone standing at the near edge of a floor zone reads as outside it,
while someone in the background reads as inside. Feet are where the person
actually is.

**So draw zone polygons on the floor**, tracing the walkable area — not around
a standing body. This is the single most common cause of a correct-looking zone
producing empty data.

`anchor_x_norm` / `anchor_y_norm` are the same point divided by frame size, so
they stay comparable when a camera's resolution changes and can be overlaid on
any frame. Use the normalized pair for heatmaps, the pixel pair for drawing on
a specific snapshot.

### `zone` always has a value

| Value | Means |
|---|---|
| a zone name | The anchor fell inside that polygon |
| `full_frame` | The camera declared `zones: "*"` — no subdivision |
| `unzoned` | Zones are defined and the anchor fell outside all of them |

Zones are evaluated in config order and **the first match wins**, so a
frame-wide zone listed first swallows every zone after it.

A large `unzoned` share is a coverage gap: adjacent polygons that do not quite
touch leave a seam, and everything landing in it is tagged `unzoned` while
tracking works perfectly. Check it before trusting zone analytics:

```sql
SELECT zone, COUNT(*),
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM   <schema>.detections
WHERE  ts > now() - INTERVAL '1 day'
GROUP  BY zone ORDER BY 2 DESC;
```

### `ts` is capture time

Taken when the frame was read, before inference. So it stays correct through a
network outage — the device buffers locally and replays hours later, and the
rows still carry the moment they happened. Never order by `id` as a proxy for
time.

---

## `detections` vs `notifications`

Both come from the same rule evaluation under different conditions.

| | `detections` | `notifications` |
|---|---|---|
| Written when | any enabled rule matched | that rule also has `notify: true`, cleared its cooldown, and sets `notifications_table` |
| Volume | every object, every frame | deduplicated by cooldown |
| Answers | "what happened" | "what needed attention" |

**Cooldown is keyed on `(rule, camera, zone)` — not on `track_id`.** Two people
entering the same zone inside the cooldown window produce **one** notification.
That makes `notifications` structurally unable to count people. Always:

```sql
COUNT(DISTINCT track_id) FROM detections     -- correct
COUNT(*)                 FROM notifications  -- counts alerts, not people
```

The **webhook** is a third path, separate from both: live-only, broadcast to the
frontend over WebSocket, never persisted and never retried. A webhook outage
loses live alerts, not stored rows.

---

## Querying it well

### Count identities, not detections

A person standing still for a minute produces hundreds of rows. That number
measures GPU throughput, not footfall.

```sql
-- unique people per zone, today
SELECT zone, COUNT(DISTINCT track_id) AS people
FROM   <schema>.detections
WHERE  "class" = 'person' AND ts::date = current_date
GROUP  BY zone ORDER BY people DESC;
```

### Dwell time

One person's stay in one zone is the span of their rows there:

```sql
SELECT zone, track_id,
       EXTRACT(EPOCH FROM (MAX(ts) - MIN(ts))) AS dwell_seconds
FROM   <schema>.detections
WHERE  "class" = 'person' AND ts > now() - INTERVAL '1 day'
GROUP  BY zone, track_id
HAVING COUNT(*) > 3;
```

`HAVING COUNT(*) > 3` drops one-frame flickers, which otherwise report 0 seconds
and drag any average down. Note that the *average* of this is a second
aggregation — see the widget-engine limits below.

### Zone-to-zone movement

Same camera only, since `track_id` does not cross cameras. A self-join on
consecutive zone changes gives transitions:

```sql
SELECT a.zone AS from_zone, b.zone AS to_zone, COUNT(*) AS moves
FROM   <schema>.detections a
JOIN   <schema>.detections b
       ON  a.track_id  = b.track_id
       AND a.camera_id = b.camera_id
       AND b.ts > a.ts
       AND b.zone <> a.zone
WHERE  a.ts > now() - INTERVAL '1 day'
GROUP  BY 1, 2 ORDER BY moves DESC;
```

### What the widget engine cannot do

A widget's `source` compiles to a **single flat `SELECT`**:

- `from` must be a bare table name in the schema — subqueries and views are rejected
- one `GROUP BY` level only, so *the average of a per-person dwell* is not expressible
- `SELECT`, `UNION` and similar keywords are blocked in every fragment
- `join` on real tables is supported, including a self-join

These are injection defences, not gaps. When a metric needs two aggregation
stages, either accept the single-stage approximation or compute it in the
backend — do not work around the filter.

---

## Volume

Four cameras at ~10 fps with people present is on the order of **1M rows/day**
in `detections`. Two consequences:

**Indexes are mandatory**, not tuning. Without them every widget sequential-scans
and the dashboard degrades as the table grows — fine for a week, then unusable.
The seven required indexes are in
[DEPLOYMENT § Stage 2](DEPLOYMENT.md#indexes).

**There is currently no retention policy.** The table grows without bound. Add
one before the deployment matures — monthly partitions on `ts`, or a scheduled
delete — and decide it deliberately rather than discovering it at disk-full.

`notifications` is orders of magnitude smaller: cooldown makes it an event log,
not a stream.

---

<div align="center">
<br/>

**[Docs index](README.md)** · **[Configuration](CONFIGURATION.md)** · **[Deployment](DEPLOYMENT.md)**

<br/>
</div>
