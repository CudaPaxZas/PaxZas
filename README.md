# Paxzas CUDA Analyzer

Static analysis of NVIDIA CUDA kernels directly inside VS Code — **no Python, no network, no GPU required**. Point it at a PTX or SASS file and get occupancy, memory posture, bottleneck diagnosis, and instruction mix instantly, across every GPU architecture from Volta to Blackwell.

---

## Features

### Multi-Architecture What-if Analysis
Every analysis runs against **all supported GPU presets in parallel**. A dropdown in the panel lets you switch results between architectures instantly — no re-running needed. Presets derived from actual SASS binary targets are marked as **native**; all others are **what-if estimates** using the same instruction profile with different SM limits.

**Native detection now compares compute-capability keys, not literal SM numbers.** A SASS dump for `sm_87` (Jetson Orin) against an `sm_86` (Ampere mobile) preset is correctly reported as **native** because both map to compute capability 8.6 — previously this was mis-flagged as a cross-arch what-if.

Low-level reference in `FEATURE_REFERENCE.md`:
- [**Occupancy Model Synthesis**](FEATURE_REFERENCE.md#part-5--occupancy-model-synthesis) (GPU-spec-driven limits and per-architecture occupancy behavior)
- [**Confidence Summary**](FEATURE_REFERENCE.md#part-7--confidence-summary) (confidence behavior across models)

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
- **Shared-memory what-if** — when shared memory is the limiting factor, an actionable "shed N bytes/block to reach next tier" recommendation is offered (parallel to the register-side margin)
- Launch parameter inference from PTX hints and optional SASS register index
- Low-level reference: [**Occupancy Model Synthesis**](FEATURE_REFERENCE.md#part-5--occupancy-model-synthesis)

### Bottleneck Diagnosis
- Fuses memory posture, stall profile, and pattern class into a **primary and secondary bottleneck** with the firing rule that triggered it
- Four **stall profile** flags: memory dependency, memory throttle, local memory (register spill), sync overhead
- Per-bottleneck optimization suggestions ranked by impact
- Low-level reference: [**Diagnosis Layer (`diagnoseKernel`)**](FEATURE_REFERENCE.md#part-8--diagnosis-layer-diagnosekernel)

### Memory Model
- Classifies the kernel as **memory-bound** or **compute-friendly**
- Arithmetic intensity (ops/byte), reuse ratio, cache policy, load/store balance and vectorization score
- **Sub-32-bit precision aware** — `LDG.E.U16` / `LDG.E.U8` (FP16, BF16, INT8 / FP8) and their store counterparts are costed at their true 2-byte / 1-byte widths, not the legacy 4-byte fallback
- **FP64-aware FLOP proxy** — `DFMA` / `DADD` / `DMUL` are counted as FP64 arithmetic and given an extra weighting (FP64 throughput is ~1/32 of FP32 on consumer GPUs), so HPC kernels are no longer mis-classified as memory-bound
- **Cache-policy precedence** — when both `.CG` and `.CS` loads exist, the dominant policy wins (`streaming` / `L2` / `mixed`); previously a single `.CG` would mask hundreds of streaming loads
- Distinguishes global, shared, and local (spill) traffic from SASS; falls back to PTX heuristics when SASS is absent
- Low-level reference: [**Memory Model Synthesis**](FEATURE_REFERENCE.md#part-4--memory-model-synthesis) and [**Feature Fusion**](FEATURE_REFERENCE.md#part-3--feature-fusion-ptx--sass--model-inputs)

### Pattern Model
- Classifies as `tiled`, `streaming`, `reduction`, `compute_heavy`, or mixed
- Detects archetypes: `GEMM`, `CONV`, `ELEMENTWISE`, `STENCIL`, and others
- **15 micro-flags**: register spill, uncoalesced loads/stores, atomic contention, missing tensor cores, SFU-heavy, over-synchronized, FP16 scalar, warp divergence, and more
- **`over_synchronized` now fires on fully-unrolled outer loops** — when PTX shows `loops = 0` because the compiler unrolled them, SASS back-edges (`BRA <addr>` whose target precedes the issuing PC) are used as a per-iteration substitute
- **Hopper / Blackwell aware** — `BARRIER` / `BMOV` / `BSSY` / `WARPSYNC` / `ARRIVE` / `WAIT` are recognised as barriers; `UTMALDG` / `UTMASTG` (TMA bulk copies), `LDGSTS` and `CP.ASYNC` (async global→shared) are counted as global memory traffic
- **PTX-only utilisation fields are honest** — `tensor_utilization_fraction` and `productive_instruction_fraction` are reported as *undefined* (rendered as `—`) when no SASS is available, instead of a misleading hard `0`
- Warp primitive detection: shuffle, vote, and reduction patterns
- Low-level reference: [**Pattern Model Synthesis**](FEATURE_REFERENCE.md#part-6--pattern-model-synthesis) and [**Feature Fusion**](FEATURE_REFERENCE.md#part-3--feature-fusion-ptx--sass--model-inputs)

### Instruction Mix (SASS)
- Category breakdown: arithmetic, tensor, SFU, global mem, shared mem, local mem, control/sync
- **Full load/store width spectrum** — LDG.128 / .64 / .32 / **.16** (half / bf16) / **.8** (int8 / fp8); STG equivalents
- **Modern tensor / async / TMA paths** — `LDGSTS`, `CP.ASYNC`, `UTMALDG`, `UTMASTG` are tracked as global ops with sub-counters (`async_global_loads`, `tma_ops`); `LDC` (constant bank) is counted separately so global byte accounting stays honest
- Tensor core op counts (WMMA / HMMA / **WGMMA** / IMMA / BMMA)
- **Precision-specific counters** — `fp16_arith_ops` (HFMA/HADD/HMUL) **and** `fp64_arith_ops` (DFMA/DADD/DMUL/DMNMX/DSETP)
- Atomic and warp-primitive counts (shuffle / vote / match)
- **`branch` and `kernel_exit` are now distinct** — RET / EXIT no longer inflate branch density on small kernels
- **`back_edges`** — backward `BRA` jumps that close a logical loop (used by `over_synchronized` when the compiler unrolled the visible loop)
- Productive instruction fraction and tensor utilization fraction (when SASS is present)
- Low-level reference: [**SASS Features**](FEATURE_REFERENCE.md#part-2--sass-features) (especially sections 2.1-2.9)

### Roofline Chart
- Plots the kernel's arithmetic intensity against the FP32 roof and bandwidth slope for the selected GPU
- Region classification: **memory-bound** or **compute-bound** with a ridge-point marker
- Low-level reference: [**Memory Model Synthesis**](FEATURE_REFERENCE.md#part-4--memory-model-synthesis) (bytes/flops and intensity derivation)

### Raw Feature Inspection
- Side-by-side PTX vs SASS instruction counts for every extracted feature
- Rows with differing values highlighted; column headers adapt to the available data source
- Low-level reference: [**PTX Features**](FEATURE_REFERENCE.md#part-1--ptx-features), [**SASS Features**](FEATURE_REFERENCE.md#part-2--sass-features), and [**Feature Fusion**](FEATURE_REFERENCE.md#part-3--feature-fusion-ptx--sass--model-inputs)

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
