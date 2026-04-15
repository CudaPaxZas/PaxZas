# Occupancy sweep webview

This document describes the **Paxzas: Occupancy Sweep (Chart)** webview: what it shows, how it is wired to the analyzer, and where the code lives.

## Purpose

The sweep answers: **how occupancy and blocks-per-SM change as block size increases**, for fixed register usage and shared memory per block (from PTX hints, optional SASS, and your GPU preset).

A second layer answers: **why the kernel might still be slow** (memory vs compute posture, pattern risks, fused diagnosis). That layer is **kernel-wide** (instruction structure does not depend on block size), so it is computed **once** and shown as badges, an insights panel, and chart tooltips—not as separate dashboards.

## How to open it

| Entry point | Command |
|-------------|---------|
| Command Palette | **Paxzas: Occupancy Sweep (Chart)** (`paxzas.occupancySweep`) |
| Editor title bar | Graph icon (when a CUDA / PTX / SASS file is active) |
| Editor / Explorer context | Same command on supported files |

**Supported inputs:** same as the main analyzer (e.g. `.ptx` with a matched `.entry`, `.cu` with embedded PTX path, `.sass` with companion context where applicable). If launch hints cannot be merged from PTX, the sweep still runs using register defaults and optional SASS-inferred registers.

## Layout (top to bottom)

1. **Subtitle** — GPU name and resolved `regs=` / `shared=` used for the sweep.
2. **Signal badges** — Up to a few high-signal flags (e.g. register spill, poor coalescing, atomics, sync overhead, tensor-core hints). Severity drives color (error / warn / info).
3. **Info cards** — GPU, registers/thread, shared bytes/block, current launch block size (if known), current occupancy and limiter at that size, peak occupancy in the sweep.
4. **Insights panel** (right column on wide layouts) — Single “Analysis” summary:
   - Primary (and optional secondary) bottleneck tags from `diagnoseKernel`
   - Memory class and arithmetic intensity (ops/byte)
   - Pattern class and optional archetype
   - At most **three** optimisation suggestions from the diagnosis ordering
5. **Charts** (unchanged count: three panels):
   - **Warp occupancy vs block size** — Line chart; optional markers where the **limiting factor** changes; current block size highlighted when known; line tint reflects **memory class** when insights exist.
   - **Blocks per SM** — Bars colored by limiting factor; tooltips mention active signal labels when insights exist.
   - **Resource limits** — Grouped bars (sampled block sizes) for max blocks allowed under each hardware constraint (capped visually for readability).

## Tooltips

Hovering a sweep point adds context without leaving the chart:

- Occupancy chart: limiting resource label, blocks/SM, and (when available) memory class and arithmetic intensity.
- Blocks chart: limiting factor plus a short “Signals” line from the badge list.

## Data flow

```text
extension (occupancy command)
  ├─ resolveGpuSpecForAnalysis(...)     → GpuSpec
  ├─ mergeLaunchWithHints(...)          → threads (current), shared, registers
  ├─ analyze(...) once                  → memory, pattern, diagnosis
  │     └─ buildSignals(pattern, memory) → KernelSignal[]
  └─ sweepBlockSizes(shared, regs, spec, currentThreads, insights)
        └─ computeOccupancyBreakdown per block size (warp step → smMaxThreads)
```

- **Per sweep point:** occupancy, blocks/SM, per-limit block counts, limiting factor.
- **Once per open:** `KernelInsights` (memory summary, pattern summary, diagnosis tags and top suggestions, signals).

## Files

| Path | Role |
|------|------|
| [`src/extension.ts`](src/extension.ts) | Registers `paxzas.occupancySweep`; loads text, runs `analyze` + `sweepBlockSizes`, opens webview. |
| [`src/analyzer/occupancy_sweep.ts`](src/analyzer/occupancy_sweep.ts) | `SweepPoint`, `SweepResult`, `KernelInsights`, `KernelSignal`, `buildSignals`, `sweepBlockSizes`. |
| [`src/webview/occupancyPanel.ts`](src/webview/occupancyPanel.ts) | `WebviewPanel`, CSP nonce, loads `media/occupancy.html`, `postMessage(sweep)`. |
| [`media/occupancy.html`](media/occupancy.html) | Markup, styles, Chart.js wiring, tooltip and insights rendering. |
| [`media/chart.min.js`](media/chart.min.js) | Vendored Chart.js UMD (no runtime network). |

## Security

The webview uses a strict **Content-Security-Policy**: scripts only with a per-load nonce; Chart and inline logic load from the extension `media/` folder via `asWebviewUri`. No external fetch.

## Related reading

- Pipeline and feature semantics: [`FEATURE_REFERENCE.md`](FEATURE_REFERENCE.md)
- General extension usage: [`README.md`](README.md)
