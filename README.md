# Paxzas CUDA Analyzer

Static analysis of NVIDIA CUDA kernels directly inside VS Code — **no Python, no network, no GPU required**. Point it at a PTX or SASS file and get occupancy, memory posture, bottleneck diagnosis, and instruction mix instantly, across every GPU architecture from Volta to Blackwell.

---

## Features

### Multi-Architecture What-if Analysis
Every analysis runs against **all supported GPU presets in parallel**. A dropdown in the panel lets you switch results between architectures instantly — no re-running needed. Presets derived from actual SASS binary targets are marked as **native**; all others are **what-if estimates** using the same instruction profile with different SM limits.

Low-level reference in `FEATURE_REFERENCE.md`:
- [**Part 5 — Occupancy Model Synthesis**](FEATURE_REFERENCE.md#part-5--occupancy-model-synthesis) (GPU-spec-driven limits and per-architecture occupancy behavior)
- [**Part 7 — Confidence Summary**](FEATURE_REFERENCE.md#part-7--confidence-summary) (confidence behavior across models)

Supported architectures:

| Preset | Architecture | SM |
|--------|--------------|----|
| `a100` | Ampere A100 | sm_80 |
| `ampere-like-default` | Ampere generic | sm_80 |
| `rtx-4090` | Ada Lovelace | sm_89 |
| `h100-sxm` | Hopper H100 SXM | sm_90 |
| `h100-pcie` | Hopper H100 PCIe | sm_90 |
| `h200` | Hopper H200 | sm_90 |
| `b200` | Blackwell B200 | sm_100 |
| `gb200` | Blackwell GB200 NVL | sm_100 |
| `blackwell-consumer-default` | Blackwell consumer | sm_120 |

### Occupancy Model
- Computes warp occupancy, blocks per SM, and the **limiting factor** (registers, shared memory, threads, or block limit) for any block size
- **Sweep charts** — occupancy, blocks/SM, and per-constraint resource limits plotted across all valid block sizes
- **Waste metrics** — unused threads, warps, registers, and shared bytes per SM at the current configuration
- Register what-if: how many fewer registers to reach the next occupancy tier
- Launch parameter inference from PTX hints and optional SASS register index
- Low-level reference: [**Part 5 — Occupancy Model Synthesis**](FEATURE_REFERENCE.md#part-5--occupancy-model-synthesis)

### Bottleneck Diagnosis
- Fuses memory posture, stall profile, and pattern class into a **primary and secondary bottleneck** with the firing rule that triggered it
- Four **stall profile** flags: memory dependency, memory throttle, local memory (register spill), sync overhead
- Per-bottleneck optimization suggestions ranked by impact
- Low-level reference: [**Part 8 — Diagnosis Layer (`diagnoseKernel`)**](FEATURE_REFERENCE.md#part-8--diagnosis-layer-diagnosekernel)

### Memory Model
- Classifies the kernel as **memory-bound** or **compute-friendly**
- Arithmetic intensity (ops/byte), reuse ratio, cache policy, load/store balance and vectorization score
- Distinguishes global, shared, and local (spill) traffic from SASS; falls back to PTX heuristics when SASS is absent
- Low-level reference: [**Part 4 — Memory Model Synthesis**](FEATURE_REFERENCE.md#part-4--memory-model-synthesis) and [**Part 3 — Feature Fusion**](FEATURE_REFERENCE.md#part-3--feature-fusion-ptx--sass--model-inputs)

### Pattern Model
- Classifies as `tiled`, `streaming`, `reduction`, `compute_heavy`, or mixed
- Detects archetypes: `GEMM`, `CONV`, `ELEMENTWISE`, `STENCIL`, and others
- **15 micro-flags**: register spill, uncoalesced loads/stores, atomic contention, missing tensor cores, SFU-heavy, over-synchronized, FP16 scalar, warp divergence, and more
- Warp primitive detection: shuffle, vote, and reduction patterns
- Low-level reference: [**Part 6 — Pattern Model Synthesis**](FEATURE_REFERENCE.md#part-6--pattern-model-synthesis) and [**Part 3 — Feature Fusion**](FEATURE_REFERENCE.md#part-3--feature-fusion-ptx--sass--model-inputs)

### Instruction Mix (SASS)
- Category breakdown: arithmetic, tensor, SFU, global mem, shared mem, local mem, control/sync
- Vectorized load/store sub-counts (LDG.128 / LDG.64 / LDG.32, STG equivalents)
- Tensor core op counts (WMMA / HMMA)
- Atomic and warp-primitive counts
- Productive instruction fraction and tensor utilization fraction
- Low-level reference: [**Part 2 — SASS Features**](FEATURE_REFERENCE.md#part-2--sass-features) (especially sections 2.1-2.9)

### Roofline Chart
- Plots the kernel's arithmetic intensity against the FP32 roof and bandwidth slope for the selected GPU
- Region classification: **memory-bound** or **compute-bound** with a ridge-point marker
- Low-level reference: [**Part 4 — Memory Model Synthesis**](FEATURE_REFERENCE.md#part-4--memory-model-synthesis) (bytes/flops and intensity derivation)

### Raw Feature Inspection
- Side-by-side PTX vs SASS instruction counts for every extracted feature
- Rows with differing values highlighted; column headers adapt to the available data source
- Low-level reference: [**Part 1 — PTX Features**](FEATURE_REFERENCE.md#part-1--ptx-features), [**Part 2 — SASS Features**](FEATURE_REFERENCE.md#part-2--sass-features), and [**Part 3 — Feature Fusion**](FEATURE_REFERENCE.md#part-3--feature-fusion-ptx--sass--model-inputs)

---

## Commands

| Command | Description |
|---------|-------------|
| **Paxzas: Kernel Analysis** | Opens the full 10-tab analysis panel |
| **Paxzas: Analyze CUDA File with Launch Spec** | Same analysis with optional `threads=…,shared=…,regs=…` overrides |

Both commands are available from the **Command Palette**, **editor title bar**, **editor right-click**, and **Explorer right-click** on `.ptx`, `.cu`, and `.sass` files.

---

## Settings

**`paxzas.gpuPreset`** — default GPU for the capability dropdown.

`auto` detects the local GPU via `nvidia-smi` when available; named presets force a specific architecture. Regardless of this setting, all presets are always shown in the panel dropdown.

---

## Requirements

- VS Code **≥ 1.85**
- Optional: `nvidia-smi` on PATH for automatic GPU detection in `auto` mode

---

## Development

```bash
npm install
npm run compile   # or: npm run watch
npm test
```

**F5** in VS Code (with this folder open) launches an Extension Development Host.

### Build a `.vsix`

```bash
./build-vsix.sh
# or
npm run vsix
```

Install locally: **Extensions → ··· → Install from VSIX…**

---

## Repository

[github.com/CudaPaxZas/PaxZas](https://github.com/CudaPaxZas/PaxZas)
