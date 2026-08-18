<div align="center">

```
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗  
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝  
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
```

### **Documentation**

<br/>

[![Docs](https://img.shields.io/badge/Docs-5%20guides-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](.)
[![Scope](https://img.shields.io/badge/Scope-Edge%20agent-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](../README.md)
[![Status](https://img.shields.io/badge/Status-Production-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](DEPLOYMENT.md)

<br/>

> *Every guide has one subject and one owner file. Nothing is documented twice.*

</div>

---

## Start here

**New to the project?** Read [`../README.md`](../README.md) for what the system
is, then [ARCHITECTURE](ARCHITECTURE.md) for how a frame becomes a database row.

**Bringing up a device?** Go straight to [DEPLOYMENT](DEPLOYMENT.md) and follow
it top to bottom.

**Writing a dashboard query?** [DATA_MODEL](DATA_MODEL.md) has every column and,
more importantly, what each one actually means.

---

## The guides

| Guide | Answers | Read it when |
|---|---|---|
| **[ARCHITECTURE](ARCHITECTURE.md)** | How the pipeline works — backends, TensorRT, tracking, ReID, zones, the anchor point | Deciding *why* something behaves the way it does |
| **[CONFIGURATION](CONFIGURATION.md)** | What each of the 8 YAML files owns and how they reference each other | Editing any config file |
| **[DATA_MODEL](DATA_MODEL.md)** | Every column of `detections`, `notifications`, `nodes`, `dashboard_config` | Writing SQL or a widget |
| **[DEPLOYMENT](DEPLOYMENT.md)** | Backend → database → edge → dashboard, end to end | Standing up a new device or branch |
| **[TOOLS](TOOLS.md)** | Running the agent, the debug tool, the service, reading the logs | Operating or troubleshooting a device |

---

## How these fit together

```
   ../README.md            what the system is, quick start
        │
        ├── ARCHITECTURE   how it works internally
        │      │
        │      ├── CONFIGURATION   ← the knobs on those internals
        │      │        │
        │      │        └── config/config_sample/*.yaml   ← per-field detail
        │      │
        │      └── DATA_MODEL      ← what it produces
        │
        ├── DEPLOYMENT     how to stand it up
        └── TOOLS          how to operate it
```

---

## Where each fact lives

Documentation drifts when the same fact is written in two places and only one
gets updated. Each fact here has exactly one home:

| Fact | Lives in | Not in |
|---|---|---|
| What a config *field* does, its range and default | The `.sample.yaml` comment next to it | Any `.md` |
| What a config *file* owns, and its cross-file links | [CONFIGURATION](CONFIGURATION.md) | The sample files |
| A column's meaning and how to aggregate it | [DATA_MODEL](DATA_MODEL.md) | DEPLOYMENT |
| The DDL to create the tables | [DEPLOYMENT](DEPLOYMENT.md) § Stage 2 | DATA_MODEL |
| Why a backend or threshold was chosen | [ARCHITECTURE](ARCHITECTURE.md) | Code comments |
| A command you type | [TOOLS](TOOLS.md) or [DEPLOYMENT](DEPLOYMENT.md) | Both |

Cross-reference with a link instead of repeating the text.

### Updating

- A code change that alters observable behaviour updates its guide **in the same commit**.
- Adding a config field: comment it in the sample file, and only touch
  [CONFIGURATION](CONFIGURATION.md) if it creates a new cross-file reference.
- Adding a column: [DATA_MODEL](DATA_MODEL.md) *and* the DDL in
  [DEPLOYMENT](DEPLOYMENT.md) § Stage 2 — those two must agree.
- Benchmarks carry the hardware, model, and date they were measured on. Replace
  them rather than appending; a table of stale numbers is worse than none.
- A new guide gets a row in the table above and a node in the diagram.

---

<div align="center">
<br/>

**[Project README](../README.md)** · **[Config samples](../config/config_sample/)**

<br/>
</div>
