# Kernel Analysis panel — UI reference

Describes what the **Paxzas: Kernel Analysis** webview looks like and how to use it.

---

## Opening the panel

| Entry point | How |
|-------------|-----|
| Command Palette | **Paxzas: Kernel Analysis** |
| Editor title bar | Search icon when a `.ptx`, `.cu`, or `.sass` file is active |
| Editor right-click | **Paxzas: Kernel Analysis** in the context menu |
| Explorer right-click | Same, on `.ptx` / `.cu` / `.sass` files |

---

## Page header

```
Kernel Analysis: <kernel name>
<file path> — <GPU spec name>
```

Below the header, a **capability bar** is always shown:

```
GPU / Capability:  [ H100 SXM (sm_90) ✓  ▾ ]   [ SASS ✓  native ]
```

- The **dropdown** lists every supported GPU preset from Volta to Blackwell.
- Options marked `✓` were found in the SASS binary (results come from real compiled code).
- Options marked `◌` are **what-if estimates** — the same kernel instructions analyzed against different SM limits.
- The **badge** on the right shows the analysis mode for the currently selected preset:
  - `SASS ✓  native` — green, SASS was compiled for this exact SM
  - `◌  what-if` — amber, occupancy/diagnosis use this SM's limits with the existing instruction profile

Switching the dropdown immediately updates all architecture-dependent tabs and rebuilds their charts.

---

## Tab bar

```
Overview | Bottleneck | Stalls | Memory | Pattern | [Occupancy] | Capabilities | Roofline | Instruction Mix | Raw Features
```

Occupancy is the default active tab.

Tabs in **bold** update when you switch GPU capability. Tabs in plain text show instruction-level data that does not change with architecture.

---

## Overview tab

**Confidence strip** (top) — four small chips showing the model confidence for each analysis layer: Memory, Pattern, Occupancy, Diagnosis.

**Info cards grid** — a row of summary cards:

| Card | What it shows |
|------|---------------|
| GPU | Selected GPU spec name |
| Kernel | PTX `.entry` name |
| Memory class | `memory_bound` / `compute_friendly` / etc. |
| Pattern class | `tiled` / `streaming` / `reduction` / etc. |
| Occupancy | Percentage and tier (e.g. `62.5% (medium)`) |
| Primary bottleneck | Top diagnosis result |
| Archetype | When detected (e.g. `GEMM`, `CONV`) |

**Diagnosis card** (below the grid) — primary bottleneck tag with its confidence, secondary bottleneck tags, and a one-line memory insight followed by a one-line pattern insight.

---

## Bottleneck tab

Lists each detected bottleneck in order (primary first, then secondary).

For each bottleneck:
- A colored tag: **red** for primary, **amber** for secondary.
- The logic rule that caused it (e.g. `memory_bound AND stall_memory_dependency`).
- One optimization suggestion when available.

When no bottleneck is detected: a green "No dominant bottleneck — the kernel appears balanced" message.

---

## Stalls tab

**Stall profile pills** — four pills, lit red when active:

| Pill | Meaning |
|------|---------|
| Memory dep | Back-to-back global loads stall on response latency |
| Memory throttle | Bandwidth-saturating pattern |
| Local memory | Register spill I/O combined with low occupancy |
| Sync | Barrier-dominated tiled kernel |

Below the pills, each stall type has an expanded row showing whether it is active and a description of the fix.

---

## Memory tab

A metrics table with 16 rows. All values are instruction-derived and do not change with GPU selection.

Rows include: class, arithmetic intensity (ops/byte), memory pressure, FLOPs proxy, reuse ratio, reuse strength, mem/compute ratio, load/store ratio and balance, store vectorization score, cache policy, global mem source, and global/shared load and store counts.

---

## Pattern tab

**Pattern overview card** — class, archetype (if any), insight text, and a compact metric table (compute/memory ratio, shared/global ratio, sync efficiency, interleaving, vectorization scores, spill severity, tensor utilization, productive instruction fraction).

**Micro-flags** — a checklist grid. Bad flags (spill, uncoalesced, over-synchronized, etc.) show a red ✗ when active. Neutral/informational flags (streaming, sync-heavy, tensor-dominated, etc.) show a green ✓.

**Warp primitives** — three checks for warp shuffle, warp vote, and warp-shuffle reduction presence.

---

## Occupancy tab

**Info cards** (top row):

| Card | What it shows |
|------|---------------|
| Threads / Block | Launch configuration |
| Regs / Thread | Register usage |
| Shared / Block | Shared memory bytes |
| Occupancy | `%  (tier)` |
| Limiting factor | Which resource caps occupancy |
| Blocks / SM | Concurrent blocks per SM |
| Register what-if | How many fewer registers to reach the next occupancy tier |
| SM utilization | Estimated (when available) |
| Peak occupancy | Best occupancy found in the sweep, at which block size |
| Current block occ. | Occupancy at the kernel's own launch block size (when known) |

**Sources line** — shows where `threads`, `shared`, and `registers` were inferred from (PTX hints, SASS inference, defaults).

**Four charts:**

1. **Warp Occupancy vs Block Size** — Line chart. Triangle markers where the limiting factor changes. Tooltip shows the limiting resource and blocks/SM at each point.
2. **Concurrent Blocks / SM vs Block Size** — Bar chart colored by limiting factor. Purple reference line for the architectural maximum. Tooltip breaks down the per-constraint block counts.
3. **Resource Limits vs Block Size** — Grouped bars (sampled) showing the maximum blocks allowed under each constraint (threads, warps, shared mem, registers, block limit). Capped at 48 for readability.
4. **Waste Mix** — Doughnut chart of unused resources per SM at the current configuration (unused threads, warps, registers, shared bytes).

**Waste metrics table** — numeric breakdown of the four waste categories plus allocated registers and shared memory per block.

**Launch warnings** — shown when present; list of warnings from the occupancy model (e.g. block size not a warp multiple).

---

## Capabilities tab

A comparison matrix with one row per preset, always showing all supported architectures.

| Column | Content |
|--------|---------|
| Preset | Name `✓` if native in SASS; `◀` marks the currently selected preset |
| SM | Compute capability tag |
| Occupancy | `%  (tier)` |
| Limit | Limiting factor |
| Blocks/SM | Concurrent block count |
| Bottleneck | Primary bottleneck label |

Rows update when you switch the capability dropdown (selected row moves).

---

## Roofline tab

A scatter + line chart:
- **Blue line** — roofline boundary (bandwidth slope up to FP32 peak ceiling) for the selected GPU.
- **Red star** — the kernel plotted at its arithmetic intensity.

**Metrics table** to the right:

| Row | Content |
|-----|---------|
| GPU arch | Spec name |
| Peak FP32 | Approximate GFLOPS for the selected GPU |
| Peak BW | Approximate GB/s |
| Ridge point | ops/byte where roofline transitions from BW-bound to compute-bound |
| Kernel intensity | Computed arithmetic intensity |
| Region | `Memory-bound` (red) or `Compute-bound` (blue) badge |
| Memory class / FLOPs proxy / Bytes proxy / Cache policy | Supporting data |

A footnote explains that peak values are approximate top-SKU estimates.

---

## Instruction Mix tab

**Category donut** — proportion of total instructions across: Arithmetic, Tensor, SFU, Global Mem, Shared Mem, Local Mem, Ctrl/Sync. Tooltip shows count and percentage per slice. Source badge (`SASS` or `PTX only`) appears in the card title.

**Productive instruction fraction** — horizontal bar (0–100%).

**Tensor utilization fraction** — horizontal bar (0–100%).

**Detailed counts table** — sections: Global Memory (with vectorized load/store breakdown when SASS), Compute, Shared/Local (SASS only), Atomics & Warp Primitives (SASS only), Control & Sync, Efficiency.

---

## Raw Features tab

Side-by-side table of every extracted feature value.

- **Title and column headers adapt** based on what is available:
  - *PTX vs SASS Feature Fusion* — both PTX and SASS present
  - *SASS Feature Counts* — SASS-only file (PTX column relabelled *Launch Params* and shows only the resolved register count)
  - *PTX Feature Counts* — PTX-only, SASS absent
- Rows where PTX and SASS values **differ** are highlighted amber.
- The SASS column shows `—` for every row when no SASS is available.

---

## Related reading

- Analyzer feature semantics and model details: [`FEATURE_REFERENCE.md`](FEATURE_REFERENCE.md)
- General extension usage: [`README.md`](README.md)
