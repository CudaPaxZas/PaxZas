# Feature Reference: PTX & SASS Extraction → Model Synthesis

This document is the single authoritative reference for every feature the PaxZas
analyser extracts from PTX and SASS files, the exact extraction rule for each
field, and how the raw features are combined ("synthesized") into the inputs
consumed by the three analytical models: **Memory**, **Occupancy**, and **Pattern**.

---

## Part 1 — PTX Features (`PtxInstructionFeatures`)

PTX is NVIDIA's intermediate representation — architecture-independent, optimiser-
friendly, but not the actual hardware instructions.  Features are extracted with
regex scans over the kernel body text.

### Extracted Fields

| Field | Type | Extraction Rule | PTX Pattern Matched |
|-------|------|-----------------|---------------------|
| `global_loads` | `number` | Count `ld.global` occurrences | `/\bld\.global\b/g` |
| `global_stores` | `number` | Count `st.global` occurrences | `/\bst\.global\b/g` |
| `fma` | `number` | Count FMA instructions | `/\bfma\./g` |
| `add` | `number` | Count stand-alone `add.*` instructions; compound sub-opcodes like `red.add` and `atom.add` are excluded | `/(?<![A-Za-z0-9_.])add\./g` |
| `mul` | `number` | Count stand-alone `mul.*` instructions; compound sub-opcodes like `shfl.idx` variants are excluded | `/(?<![A-Za-z0-9_.])mul\./g` |
| `barrier` | `number` | Count `bar.sync` instructions | `/\bbar\.sync\b/g` |
| `reg_decl_lines` | `number` | Count `.reg` declaration lines | `/^\s*\.reg\b/` |
| `branches` | `number` | Count `bra` opcodes (all variants) | `/\bbra(?:\.[A-Za-z0-9_]+)*\b/g` |
| `loops` | `number` | Count backward branch edges (label appears before the `bra` target) | Two-pass: labels → line numbers, then `bra` targets with label line < current line. Label grammar accepts LLVM-style and non-LLVM-style names (e.g. `L0:`, `BB0_1:`, `.Ltmp0:`, `$foo.bar:`). |

### Derived Heuristics (computed from raw fields)

| Derived Value | Formula | Purpose |
|---------------|---------|---------|
| `flopsHeuristic` | `fma × 2 + add + mul` | Estimated FLOPs (FMA counts as 2 ops) |
| `bytesHeuristic` | `(global_loads + global_stores) × 4` | Estimated bytes moved (assumes 4 B/op) |

### Limits & Gaps

- **No shared memory counts** — PTX does not expose `lds`/`sts` instruction counts in a reliable way, so `shared_loads = 0` and `shared_stores = 0` when PTX-only.
- **No byte-width precision** — all loads and stores are assumed to be 4 bytes.
- **No tensor-core visibility** — `mma.*` instructions are not parsed.
- **No cache-policy information** — `.cs`, `.cg` modifiers are ignored.
- **No spill detection** — `ld.local`/`st.local` equivalents are not tracked.
- **Register count is structural** — `reg_decl_lines` counts `.reg` declaration lines, not the actual maximum register index used.

---

## Part 2 — SASS Features (`SassInstructionFeatures`)

SASS is the final hardware assembly produced by `ptxas`.  A single forward scan
over the `cuobjdump --dump-sass` output processes every instruction line.
Predicate guards (`@P0`, `@!P1`) are stripped before opcode matching so that
predicated branches are never silently dropped.
Register-index extraction intentionally scans the raw line with `R\d+`; this is
robust for both plain operands (`R12`) and indexed constant-bank operands
(`c[0x0][R4]`) while ignoring numeric literals (`0x10`).

### Extraction Entry Point

```
extractSassFeatures(sassText, kernelSubstring?)
  → [matchedKernelName, SassInstructionFeatures]
```

Stops at the next `Function:` header so only one kernel section is processed.

`inferRegistersPerThreadFromSass(sassText, kernelSubstring?, precomputed?)` maps
`max_register_index + 1` to a register count.  When `precomputed` is the
`SassInstructionFeatures` object already returned from `extractSassFeatures`, the
helper skips a second full scan (I7).

---

### 2.1 Global Memory

| Field | Type | SASS Opcodes | Notes |
|-------|------|-------------|-------|
| `global_loads` | `number` | `LDG.*`, `LDGSTS.*`, `CP.ASYNC.*`, `UTMALDG.*` | Total global loads regardless of width — **includes** Ampere `LDGSTS` and Hopper / Blackwell `CP.ASYNC` (async global → shared) and `UTMALDG` (TMA bulk load) |
| `global_stores` | `number` | `STG.*`, `UTMASTG.*` | Total global stores regardless of width — includes Hopper `UTMASTG` (TMA bulk store) |
| `async_global_loads` | `number` | `LDGSTS.*`, `CP.ASYNC.*` | Sub-share of `global_loads` issued via the async copy pipeline (bypasses the register file). |
| `tma_ops` | `number` | `UTMALDG.*`, `UTMASTG.*` | Hopper / Blackwell Tensor-Memory-Accelerator bulk transfer count. |
| `const_loads` | `number` | `LDC.*` | Constant-bank reads (broadcast, cached at L1). **Not counted** in `global_loads` — kept separate so global byte accounting is honest. |

#### Load width sub-counters

| Field | Width | Increment when |
|-------|-------|----------------|
| `ldg_128` | 16 B | `.128` in opcode |
| `ldg_64` | 8 B | `.64`, `.U64`, `.S64`, `.F64` in opcode |
| `ldg_32` | 4 B | `.32`, `.U32`, `.S32`, `.F32` in opcode |
| `ldg_16` | 2 B | `.U16`, `.S16`, `.F16`, or `.16` in opcode (FP16 / BF16 / INT16) |
| `ldg_8` | 1 B | `.U8`, `.S8`, or `.8` in opcode (INT8 / FP8 / `char`) |

> The 16-bit and 8-bit buckets are matched **before** the bare-numeric `.8` test so that `.U16` is not mis-routed to `ldg_8`.

#### Store width sub-counters

| Field | Width | Increment when |
|-------|-------|----------------|
| `stg_128` | 16 B | `.128` in opcode |
| `stg_64` | 8 B | `.64`, `.U64`, `.S64`, `.F64` in opcode |
| `stg_32` | 4 B | `.32`, `.U32`, `.S32`, `.F32` in opcode |
| `stg_16` | 2 B | `.U16`, `.S16`, `.F16`, or `.16` in opcode |
| `stg_8` | 1 B | `.U8`, `.S8`, or `.8` in opcode |

Unknown-width loads/stores (no width qualifier) fall back to the `defaultBytesPerOp`
(4 B) when bytes are computed.

---

### 2.2 Cache Policy

| Field | Type | SASS Opcode Modifier | Hardware Meaning |
|-------|------|---------------------|-----------------|
| `cg_loads` | `number` | `.CG` on `LDG` | Cache Global — routes through L2 normally |
| `cs_loads` | `number` | `.CS` on `LDG` | Cache Streaming — bypasses L1/L2, used for streaming non-reused data |

---

### 2.3 Shared Memory

| Field | Type | SASS Opcodes | Notes |
|-------|------|-------------|-------|
| `shared_loads` | `number` | `LDS.*` | All shared-memory loads (any width) |
| `shared_stores` | `number` | `STS.*` | All shared-memory stores (any width) |

PTX cannot reliably report these; SASS is the only reliable source.

---

### 2.4 Local Memory (Register Spill)

| Field | Type | SASS Opcode | What it means |
|-------|------|------------|---------------|
| `local_loads` | `number` | `LDL.*` | Compiler reloaded a value from per-thread local memory — register was spilled |
| `local_stores` | `number` | `STL.*` | Compiler saved a live value to per-thread local memory — about to exhaust registers |

Any non-zero value is evidence that the register file was exhausted and the compiler
evicted live values to L1/L2/DRAM-backed local memory at 100+ cycle latency per access.

---

### 2.5 Compute

#### Scalar arithmetic

| Field | Type | SASS Opcodes | Notes |
|-------|------|-------------|-------|
| `arithmetic_ops` | `number` | `FFMA`, `FADD`, `FMUL`, `IADD`, `IMAD`, `IMUL`, `HFMA`, `HADD`, `HMUL`, **`DFMA`**, **`DADD`**, **`DMUL`**, **`DMNMX`**, **`DSETP`** | All scalar FP/int arithmetic (FP16 and FP64 scalar are subsets) |
| `integer_ops` | `number` | `IADD`, `IMAD`, `IMUL`, `IMNMX`, `ISCADD`, `ISET`, `ICMP`, `IABS`, `INEG`, `IAND`, `IOR`, `IXOR`, `ISHL`, `ISHR` | Integer/address/control-heavy subset used by `fp_to_int_ratio` |
| `fp16_arith_ops` | `number` | `HFMA`, `HADD`, `HMUL` | Scalar FP16 subset — **also** counted in `arithmetic_ops` |
| `fp64_arith_ops` | `number` | `DFMA`, `DADD`, `DMUL`, `DMNMX`, `DSETP` | Scalar FP64 subset — **also** counted in `arithmetic_ops`. Tracked separately so `flops_proxy` can apply the FP64 throughput penalty (1/32 of FP32 on consumer GPUs); without this counter, CFD / molecular-dynamics kernels showed `compute_ops ≈ 0` and were always classified as `memory_bound`. |

#### Tensor core

| Field | Type | SASS Opcodes | Notes |
|-------|------|-------------|-------|
| `tensor_ops` | `number` | `MMA`, `HMMA`, `IMMA`, `BMMA`, `WGMMA`, `WMMA` | All tensor-core instructions regardless of data type |
| `wmma_ops` | `number` | `HMMA`, `WGMMA`, `WMMA` | FP tensor-core subset only — excludes `IMMA` (integer) and `BMMA` (binary) |

#### Special Function Unit

| Field | Type | SASS Opcode | CUDA intrinsic |
|-------|------|-------------|----------------|
| `sfu_ops` | `number` | `MUFU.*` | Maps to `sinf`, `cosf`, `expf`, `logf`, `rcpf`, `rsqrtf`, `sqrtf`; throughput = ¼ of FP32 ALU |

---

### 2.6 Synchronisation & Control Flow

| Field | Type | SASS Opcodes |
|-------|------|-------------|
| `barrier` | `number` | `BAR`, `DEPBAR`, `MEMBAR`, **`BARRIER`** (cluster-wide), **`BMOV`**, **`BSSY`** (Hopper split-barrier setup), **`WARPSYNC`**, **`ARRIVE`**, **`WAIT`** (mbarrier — Hopper / Blackwell cooperative groups). |
| `branch` | `number` | `BRA`, `JMP`, `SSY`, `SYNC` — **including** predicated forms `@P0 BRA`. **Excludes** `RET` / `EXIT` (counted in `kernel_exit`). |
| `kernel_exit` | `number` | `RET`, `EXIT` — split out from `branch` so per-kernel exits don't inflate branch density on small kernels. |
| `back_edges` | `number` | `BRA <addr>` whose target hex address is **less than** the issuing PC (a backward jump that closes a loop). Used by the pattern model's `over_synchronized` check as a substitute for PTX `loops` when the compiler fully unrolled the visible loop. |

---

### 2.7 Atomics

| Field | Type | SASS Opcodes | Notes |
|-------|------|-------------|-------|
| `atomic_ops` | `number` | `ATOM`, `ATOMS`, `RED` | **Total** atomic instruction count — global (`ATOM`/`RED`) + shared (`ATOMS`). Shared-memory atomics serialize at the shared-bank level; they do NOT cause L2 contention. |
| `global_atomic_ops` | `number` | `ATOM`, `RED` (not `ATOMS`) | **Global-only** atomics that serialize through the L2 cache partition. This is the correct numerator for the `atomic_contention_risk` signal. `ATOMS` is excluded. |

---

### 2.8 Aggregate & Streaming

| Field | Type | Description |
|-------|------|-------------|
| `total_instructions` | `number` | Total instructions processed in the kernel section |
| `max_register_index` | `number` | Highest `RN` register index seen (used to infer actual register count = `max_register_index + 1`) |
| `instruction_sequence` | `string[]` | Ordered opcode list (capped at 65 536) for sequence-pattern analysis |
| `stream_interleave_score` | `number` | 0–1 measure of how frequently loads and compute instructions alternate (high = good overlap) |
| `stream_max_consecutive_loads` | `number` | Longest run of consecutive load instructions (high = potential stall chain) |

---

### 2.9 Warp-Level Primitives

| Field | Type | SASS Opcodes | Notes |
|-------|------|-------------|-------|
| `warp_shuffle_ops` | `number` | `SHFL.*` | Any warp-shuffle variant (IDX, UP, DOWN, BFLY) — warp communicates register values without shared memory |
| `warp_vote_ops` | `number` | `VOTE.*`, `MATCH.*` | Warp-vote predicates and match-any/all — warp executes a collective boolean reduce |

---

## Part 3 — Feature Fusion: PTX + SASS → Model Inputs

When SASS is available it takes precedence because it is exact hardware-level data;
PTX is used as a fallback heuristic.  The selection rule applied in every model is:

> **"If SASS was provided (`sassFeatures !== undefined`), use its count for every group — even when the count is zero.  Only fall back to PTX when SASS was not provided at all."**

This is the correct rule after the B5 fix.  The previous implementation used `sassCount > 0 ? sass : ptx`, which silently substituted a stale PTX value whenever SASS legitimately reported zero for a group (e.g. zero global loads in a pure shared-memory kernel).  That corrupted `globalOps`, `computeOps`, `barriers`, and `branches` in exactly the cases where the data was most trustworthy.

The only exception is **`loops`**, which always comes from PTX backward-branch-edge counting because SASS has no loop-edge semantics at the instruction level.

The table below shows every fused quantity, which source wins, and which model
consumes it.

| Fused Quantity | SASS Source | PTX Fallback | Consumers |
|----------------|-------------|-------------|-----------|
| `global_loads` | `sass.global_loads` | `ptx.global_loads` | Memory, Pattern |
| `global_stores` | `sass.global_stores` | `ptx.global_stores` | Memory, Pattern |
| `shared_loads` | `sass.shared_loads` | 0 (PTX unknown) | Memory, Pattern |
| `shared_stores` | `sass.shared_stores` | 0 (PTX unknown) | Memory, Pattern |
| `barriers` | `sass.barrier` | `ptx.barrier` | Pattern |
| `branches` | `sass.branch` | `ptx.branches` | Pattern |
| `loops` | — | `ptx.loops` (always PTX — SASS has no loop semantics) | Pattern |
| `compute_ops` | `sass.arithmetic_ops + sass.tensor_ops` | `ptx.fma + ptx.add + ptx.mul` | Memory, Pattern |
| `bytes_moved` | `estimateSassBytes()` (typed widths, **including 2-byte and 1-byte sub-32-bit loads/stores**) | `(loads + stores) × 4` | Memory |
| `flops_proxy` | `scalar×2 + wmma×512 + intMMA×64 + sfu×4 + fp64×14`; if that sum is **0** but `integer_ops > 0`, falls back to `integer_ops×2` (I9 — ISCADD-style ALU not in `arithmetic_ops`) | `fma×2 + add + mul` | Memory |
| `registers_per_thread` | `sass.max_register_index + 1` | `ptx.maxnreg` hint or launch param | Occupancy |
| `threads_per_block` | — | `ptx.maxntid` hint or launch param | Occupancy |
| `shared_mem_per_block` | — | `ptx.static_shared` (bytes, typed element size applied) or launch param | Occupancy |
| `cache_policy` | dominant policy wins: `cs > cg → "streaming"`, `cg > cs → "L2"`, equal non-zero counts → `"mixed"`, both zero → `null` | `null` | Memory |
| `tensor_ops` | `sass.tensor_ops` | 0 | Pattern |
| `integer_ops` | `sass.integer_ops` | 0 | Pattern (`fp_to_int_ratio`) |
| `wmma_ops` | `sass.wmma_ops` | 0 | Pattern, Memory |
| `local_loads` | `sass.local_loads` | 0 | Pattern |
| `local_stores` | `sass.local_stores` | 0 | Pattern |
| `atomic_ops` | `sass.atomic_ops` | 0 | Pattern (total serialization count) |
| `global_atomic_ops` | `sass.global_atomic_ops` | 0 | Pattern (`atomic_contention_risk` numerator) |
| `sfu_ops` | `sass.sfu_ops` | 0 | Pattern, Memory |
| `fp16_arith_ops` | `sass.fp16_arith_ops` | 0 | Pattern |
| `ldg_128/64/32/16/8` | `sass.ldg_128/64/32/16/8` | 0 | Pattern (vectorization_score uses ≥32-bit buckets), Memory (all five buckets feed `bytes_moved`) |
| `stg_128/64/32/16/8` | `sass.stg_128/64/32/16/8` | 0 | Memory (store byte estimate + store_vectorization_score), Pattern (store-side coalescing/vectorization) |
| `fp64_arith_ops` | `sass.fp64_arith_ops` | 0 | Memory (FP64 FLOP-proxy weighting), Pattern |
| `const_loads` | `sass.const_loads` | 0 | (informational — kept out of `bytes_moved`) |
| `async_global_loads` | `sass.async_global_loads` | 0 | Pattern |
| `tma_ops` | `sass.tma_ops` | 0 | Pattern |
| `kernel_exit` | `sass.kernel_exit` | 0 | (informational — split off `branch`) |
| `back_edges` | `sass.back_edges` | 0 | Pattern (`over_synchronized` fallback when PTX `loops = 0`) |
| `stream_interleave_score` | `sass.stream_interleave_score` | 0 | Pattern (interleaving) |
| `max_consecutive_loads` | `sass.stream_max_consecutive_loads` | 0 | Pattern (interleaving) |
| `warp_shuffle_ops` | `sass.warp_shuffle_ops` | 0 | Pattern (F2/F4 warp primitives) |
| `warp_vote_ops` | `sass.warp_vote_ops` | 0 | Pattern (F3 warp primitives) |

---

## Part 4 — Memory Model Synthesis

**Entry point:** `analyzeMemory(ptxFeatures, sassFeatures?, bytesPerMemOp=4)`

### Step 1 — Byte estimation

```
knownLdg   = ldg_128 + ldg_64 + ldg_32 + ldg_16 + ldg_8
unknownLdg = max(0, global_loads − knownLdg)

knownStg   = stg_128 + stg_64 + stg_32 + stg_16 + stg_8
unknownStg = max(0, global_stores − knownStg)

bytes_moved =
    ldg_128 × 16 + ldg_64 × 8 + ldg_32 × 4 + ldg_16 × 2 + ldg_8 × 1 + unknownLdg × 4
  + stg_128 × 16 + stg_64 × 8 + stg_32 × 4 + stg_16 × 2 + stg_8 × 1 + unknownStg × 4

  PTX fallback: bytes_moved = (global_loads + global_stores) × 4
```

The 2-byte and 1-byte buckets matter for FP16 / BF16 / INT8 / FP8 kernels (CUTLASS, Flash-Attention, quantised inference): without them, a kernel doing 16-bit traffic was costed at 4 B per load and arithmetic intensity reported 50 % low.

### Step 2 — FLOP estimation

```
scalar_flops      = arithmetic_ops × 2
tensor_fp_flops   = wmma_ops × 512               (HMMA/WGMMA/WMMA — 16×16×16 MAC)
tensor_int_flops  = (tensor_ops − wmma_ops) × 64 (IMMA/BMMA conservative floor)
sfu_flops         = sfu_ops × 4                  (MUFU ≈ 4 float-op equivalents)
fp64_penalty      = fp64_arith_ops × 14          (extra weight on top of the +2 already in scalar_flops; total ≈ 16× per FP64 op)

flops_proxy = scalar_flops + tensor_fp_flops + tensor_int_flops + sfu_flops + fp64_penalty

  PTX fallback: flops_proxy = fma × 2 + add + mul
  Tie-break:    flops = max(sass_flops_proxy, ptx_flops_proxy)
```

> The FP64 penalty reflects the relative cost of the FP64 pipeline (≈1/32 throughput of FP32 on consumer GPUs), so a single `DFMA` is treated as roughly 16× the work of a single `FFMA` for intensity purposes. This is what stops CFD / molecular-dynamics kernels from being flagged as `memory_bound` despite spending the bulk of their time in FP64 ALU.

### Step 3 — Derived ratios

| Output Field | Formula |
|-------------|---------|
| `arithmetic_intensity_ops_per_byte` | `flops / bytes_moved` (∞ when bytes = 0) |
| `reuse_ratio` | `shared_loads / global_loads` |
| `reuse_strength` | `shared_loads / global_mem_ops` |
| `mem_compute_ratio` | `bytes_moved / flops` |
| `store_vectorization_score` | `(stg_128×4 + stg_64×2 + stg_32×1) / (totalStgTyped × 4)` (0 when `< 4` typed stores) |
| `load_store_ratio` | `global_loads / (global_stores + 1)` |
| `load_store_balance` | `"read_dominated"` if ratio > 4.0; `"write_dominated"` if ratio < 0.5; else `"balanced"` |

### Step 4 — Classification

| Class | Condition (evaluated in order) |
|-------|-------------------------------|
| `compute_friendly` | `bytes_moved = 0` with `flops > 0` (pure compute) |
| `memory_bound` | `isStreaming && intensity < 0.5` |
| `memory_bound` | `intensity < 0.5 && global_mem_ops ≥ 4` |
| `reuse_optimized` | `reuse_ratio > 2.0 && reuse_strength > 1.0` |
| `compute_friendly` | `intensity > 2.0` |
| `balanced` | (all other cases) |

`isStreaming` = `cs_loads > 0` from SASS cache policy.

#### `cache_policy` precedence rule

When both `.CG` and `.CS`-flagged loads exist, the dominant policy wins:

| `cs_loads` vs `cg_loads` | Reported `cache_policy` |
|--------------------------|------------------------|
| `cs > cg > 0` | `"streaming"` |
| `cg > cs > 0` | `"L2"` |
| `cs == cg > 0` | `"mixed"` |
| `cs > 0`, `cg == 0` | `"streaming"` |
| `cg > 0`, `cs == 0` | `"L2"` |
| `cs == 0` and `cg == 0` | `null` |

Previously a single `.CG` load suppressed the streaming classification on a kernel with hundreds of `.CS` loads, and consumed the `+0.10` streaming confidence bump.

### Step 4b — Data-source provenance fields

`MemoryAnalysis` carries three string fields that identify where each group of metrics actually came from.  These drive the "Sources" row in the UI's Raw Features tab.

| Field | Value `"sass"` when | Value `"ptx"` / `"unknown"` when |
|-------|--------------------|---------------------------------|
| `global_mem_source` | SASS reported `global_loads + global_stores > 0` | SASS absent **or** both counts are 0 (falls back to PTX) |
| `shared_mem_source` | SASS reported `shared_loads + shared_stores > 0` | SASS absent or both counts are 0 → `"unknown"` (PTX cannot report shared ops) |
| `compute_source` | SASS object was passed to `analyzeMemory` (regardless of arithmetic op count) | SASS was not passed at all → `"ptx"` |

> **Why `compute_source` uses presence, not count:** A pure-memory or control-flow SASS kernel legitimately has zero arithmetic operations.  Reporting `"ptx"` in that case would be incorrect — no PTX compute data was consulted.  The rule is: *if SASS was the source, say so*, even when its arithmetic count is zero.

### Step 4c — Legacy `heuristicBottleneck` (I11)

`heuristicBottleneck(ptxFeatures, registers, …, sassFeatures?)` mirrors the memory
model when SASS is present: FLOPs use `sassFlopsProxyFromFeatures`, and when SASS
reports global loads or stores, bytes use `sassBytesProxyFromFeatures` and
`global_mem_ops` uses the SASS global-op tally.  This keeps the quick
memory-bound / compute-bound label aligned with `MemoryAnalysis` on SASS-rich
inputs.  Notes added: `sass_flops_proxy` (when SASS FLOPs exceed PTX FLOPs) and
`sass_global_bytes_proxy` (when SASS global ops are present).

### Step 5 — Confidence scoring

| Condition | Effect |
|-----------|--------|
| Base | `+0.60` |
| SASS available | `+0.25` |
| `global_mem_ops = 0` AND SASS absent | **hard override** to `0.30` — PTX zero is unreliable; hard to classify |
| `global_mem_ops = 0` AND SASS present | no override — SASS confirmed zero; high confidence kept |
| `reuse_ratio > 2.0` OR `intensity < 0.5` | `+0.10` |
| `isStreaming` | `+0.10` (unambiguous signal) |
| `global_mem_ops < 5` (SASS present) | `−0.10` |
| `global_mem_ops < 5` (PTX only) | `−0.20` |
| Cap | `min(result, 0.99)` |

---

## Part 5 — Occupancy Model Synthesis

**Entry point:** `analyzeKernel(threadsPerBlock, sharedMemPerBlock, registersPerThread, spec, opts?)`

Optional `opts.gridBlocks` caps `estimated_sm_utilization` when the launch has
fewer thread blocks than the occupancy-derived SM participation (I3).

The first three inputs are themselves synthesized by `mergeLaunchWithHints()` using a
priority waterfall:

### Step 1 — Launch parameter resolution (`mergeLaunchWithHints`)

| Parameter | Priority 1 (highest) | Priority 2 | Priority 3 |
|-----------|---------------------|-----------|-----------|
| `registers_per_thread` | Explicit `--launch regs=` | `ptx.maxnreg` hint (line-anchored regex — ignores `//` comments) | `sass.max_register_index + 1` |
| `threads_per_block` | Explicit `--launch threads=` | `ptx.maxntid` hint (line-anchored, same rule) | — (required) |
| `shared_mem_per_block` | Explicit `--launch shared=` | `ptx.static_shared` bytes (element count × byte-width of declared type) | 0 |
| `gridBlocks` (optional) | Explicit `--launch grid=` or VS Code `grid=…` in launch spec | — | — |

`gridBlocks` is forwarded only to `analyzeKernel` for the SM-util string; it does not change occupancy math.

Source labels (`"launch"`, `"ptx.maxntid"`, `"ptx.maxnreg"`, `"ptx.static_shared"`,
`"sass.inferred"`) are carried forward into `OccupancyModelResult.sources` for
transparency.

### Step 2 — Per-resource block limits

Each limit is independently computed then the minimum becomes `blocks_per_sm`.

| Limit | Formula |
|-------|---------|
| `blocks_by_threads` | `⌊smMaxThreads / threadsPerBlock⌋` (0 if block is too large) |
| `blocks_by_warps` | `⌊smMaxWarps / ⌈threadsPerBlock / 32⌉⌋` |
| `blocks_by_shared` | `⌊smMaxSharedMem / roundUp(sharedPerBlock, allocUnit)⌋` |
| `blocks_by_registers` | `⌊smMaxRegisters / (warpsPerBlock × roundUp(R × 32, regAllocUnitPerWarp))⌋` |
| `blocks_by_block_limit` | `smMaxBlocks` (hard architectural constant) |

> **Register allocation is per-warp**, not per-block.  The CUDA driver rounds
> each warp's register demand up to `regAllocUnitPerWarp` (256 on all modern GPUs)
> then multiplies by `warpsPerBlock`.  Using per-block rounding overstates occupancy.

### Step 3 — Occupancy and waste (`KernelAnalysis` — from `analyzeKernel()`)

> **Two result interfaces exist:**
> - `KernelAnalysis` (returned by `analyzeKernel()`) is the rich, detailed result used internally. It contains `occupancy_class`, `warnings[]`, `warp_metrics`, `waste_metrics`, `limits`, and `estimated_sm_utilization`.
> - `OccupancyModelResult` (returned by `analyzeOccupancyModel()`) is the model-facing summary. It contains `class` (not `occupancy_class`), `confidence`, `insight`, and `sources`, but has **no** `warnings[]` or `estimated_sm_utilization`.

```
blocks_per_sm       = min(all five limits above)
active_warps_per_sm = min(blocks_per_sm × warpsPerBlock, smMaxWarps)
occupancy           = active_warps_per_sm / smMaxWarps   [0.0 – 1.0]
```

**`KernelAnalysis` fields:** `occupancy_class` = `"low"` (<0.30) | `"medium"` (<0.60) | `"high"` (≥0.60); `register_pressure_margin`; `next_occupancy_class`; `next_limiting_factor`; `shared_mem_pressure_margin`; `next_occupancy_class_shared`; `next_limiting_factor_shared`; `warp_metrics`; `waste_metrics` (unused resource fractions per SM).

`register_pressure_margin` is only populated when `limiting_factor = "registers"` and a higher occupancy tier exists; it reports regs/thread to shed to reach `next_occupancy_class`.
The margin is quantized to the hardware allocation granularity `regAllocUnitPerWarp / warpSize` (8 regs/thread on modern NVIDIA), so the recommendation lines up with what `ptxas` can actually realize.

`next_limiting_factor` is set alongside `register_pressure_margin` when shedding the margin regs causes a **different resource to become the new bottleneck** (e.g. `"shared_mem"`).  The tier improvement is still achievable — the register-reduction advice remains valid — but the field signals that a second optimisation will be needed to improve occupancy further.

`shared_mem_pressure_margin` is the parallel of `register_pressure_margin` for shared-memory-limited kernels: when `limiting_factor = "shared_mem"` and a higher occupancy tier exists, it reports the **bytes/block to shed** to reach `next_occupancy_class_shared`. `next_limiting_factor_shared` flags the resource that becomes the new bottleneck after the suggested reduction.

> Together, the register and shared-memory pressure margins back two actionable warning forms in `warnings[]`: `"≤N regs/thread (add __launch_bounds__(T)) to fit 2 blocks/SM"` and `"≤N bytes/block to fit 2 blocks/SM"`.

**`OccupancyModelResult` fields:** `class` (same tier string), `confidence` (0.65–0.85), `limiting_factor`, `blocks_per_sm`, `threads_per_block`, `shared_mem_per_block`, `registers_per_thread`, `sources` (origin of each parameter), `insight`.

### Step 4 — Actionable warnings (in `KernelAnalysis.warnings[]`)

| Condition | Warning |
|-----------|---------|
| `blocks_per_sm = 0` | **"Kernel cannot launch"** — early return, no other checks |
| `threads < 128` | Poor latency hiding |
| `threads < 64` | Very low (suspicious) |
| `threads % 32 ≠ 0` | Wasted lanes |
| `warps_per_block < 4` | Poor SM utilization |
| `limiting_factor = "registers"` | `"≤N regs/thread (add __launch_bounds__(T)) to fit 2 blocks/SM"` |
| `registers_per_thread > 64` | Very high register usage |
| `limiting_factor = "shared_mem"` | `"≤N bytes/block to fit 2 blocks/SM"` |
| `occupancy < 0.4` | Poor latency hiding |

### Step 5 — Device-level utilization (in `KernelAnalysis.estimated_sm_utilization`)

When `GpuSpec.smCount` is known (from nvidia-smi or preset):

```
idealSms = round(occupancy × smCount)
displayedSms = min(idealSms, gridBlocks)   when opts.gridBlocks is a finite non‑negative integer
estimated_sm_utilization = displayedSms + " / " + smCount + " SMs active" [+ "; grid≤N blocks" when capped]
```

Note: `estimated_sm_utilization` is on `KernelAnalysis` only. `OccupancyModelResult` does not expose it directly.
Without `gridBlocks`, the string still assumes enough blocks exist to reach the occupancy-derived participation; pass `grid=` in the launch spec (or VS Code launch hints) to model small grids (I3).

**SM count name-heuristic SKU rules (H100):** The H100 has two distinct enabled SM counts depending on the physical form factor. The heuristic in `inferSmCountFromGpuName` matches in priority order:

| nvidia-smi name pattern | SM count | SKU |
|------------------------|----------|-----|
| `H100 … PCIe` | 114 | H100 PCIe |
| `H100 … NVL` | 114 | H100 NVL |
| `H100` (generic fallback) | 132 | H100 SXM5 |
| `H200` | 132 | H200 SXM |

**Additional consumer-name heuristic examples (first-match order):**

| nvidia-smi name pattern | SM count |
|------------------------|----------|
| `RTX 4080 SUPER` | 80 |
| `RTX 4080` | 76 |
| `RTX 4070 Ti SUPER` | 66 |
| `RTX 4070 Ti` | 60 |
| `RTX 4070` | 46 |

---

## Part 6 — Pattern Model Synthesis

**Entry point:** `analyzePattern(ptxFeatures, sassFeatures?)`

### Step 1 — Metric extraction (SASS wins)

| Internal Variable | Value (SASS present, even if count = 0) | Value (PTX only — SASS not provided) |
|------------------|----------------------------------------|--------------------------------------|
| `globalOps` | `sass.global_loads + sass.global_stores` | `ptx.global_loads + ptx.global_stores` |
| `sharedOps` | `sass.shared_loads + sass.shared_stores` | 0 (PTX cannot report shared ops) |
| `barriers` | `sass.barrier` | `ptx.barrier` |
| `branches` | `sass.branch` | `ptx.branches` |
| `loops` | `ptx.loops` | `ptx.loops` (always PTX — SASS has no loop-edge semantics) |
| `computeOps` | `sass.arithmetic_ops + sass.tensor_ops` | `ptx.fma + ptx.add + ptx.mul` |

### Step 2 — Derived ratios

| Ratio | Formula | Threshold used |
|-------|---------|---------------|
| `sharedToGlobal` | `sharedOps / globalOps`, **+∞** when `globalOps = 0` and `sharedOps > 0` (I1) | > 1.5 → strong reuse |
| `computeToMemory` | `computeOps / (globalOps + 1)` when `globalOps > 0`; **+∞** when `globalOps = 0` and `computeOps > 0` (I1) | > 8: tiled; > 4: compute-heavy; < 2: streaming |

Exported `shared_to_global_ratio` / `compute_to_memory_ratio` replace **+∞** with a JSON-safe sentinel (`1e12`) so webviews do not receive `null` from `JSON.stringify(Infinity)`.
| `branchDensity` | `branches / (computeOps + 1)` | > 0.10 → control-heavy |
| `branchPerMem` | `branches / (globalOps + 1)` | > 0.08 (+ low compute) → control-heavy |
| `barrierDensity` | `barriers / (computeOps + 1)` | > 0.02 → sync-heavy |
| `workPerBarrier` | `computeOps / (barriers + 1)` | > 100: efficient; > 30: moderate; else: inefficient |
| `loopDensity` | `loops / (computeOps + 1)` | > 0.05 → high-looping |
| `totalLdgTyped` | `ldg_128 + ldg_64 + ldg_32` | ≥ 4 for vectorization_score |
| `totalStgTyped` | `stg_128 + stg_64 + stg_32` | ≥ 4 for store_vectorization_score |

### Step 3 — Primary class decision tree

```
IF highBranching (branchDensity > 0.1 OR branchPerMem > 0.08 with low compute)
    → "control_heavy"

ELSE IF hasShared AND hasBarrier AND hasLoops:
    IF computeToMemory > 8.0  → "tiled"
    ELSE                      → "reduction"

ELSE IF NOT hasShared AND NOT hasBarrier:
    IF computeToMemory > 2.0  → "compute_heavy"
    ELSE                      → "elementwise"

ELSE
    → "irregular"
```

Source label: `"sass"` if SASS was available and produced non-zero global/shared
traffic; `"ptx"` otherwise.  SASS presence adds `+0.1` to confidence.

### Step 3b — Multi-kernel PTX suppression (B9)

When a PTX translation unit contains more than one `.entry` kernel and no
`kernelFilter` is specified, `extractPtxKernelsMerged` sums the instruction
counters of all kernels into a single `PtxInstructionFeatures` object and sets
`kernelCount = N` (N ≥ 2).  Loop and branch counts then cross kernel
boundaries, making density ratios unreliable for any individual kernel.

**Rule:** If `ptxFeatures.kernelCount > 1` **and** SASS was *not* provided,
the following boolean flags are forced to `false` and `multi_kernel_ptx` is
set to `true` in the `PatternResult`:

| Flag forced to `false` |
|------------------------|
| `high_looping` |
| `warp_divergence_risk` |
| `over_synchronized` |
| `complex_kernel` |

When SASS *is* present (`sassFeatures !== undefined`) it provides single-kernel
counts, so suppression does **not** apply even if `kernelCount > 1`.

### Step 4 — Micro-pattern boolean flags

| Flag | Condition |
|------|-----------|
| `high_looping` | `loopDensity > 0.05` (suppressed for multi-kernel PTX sum) |
| `sync_heavy` | `barrierDensity > 0.02` |
| `control_irregular` | `branchDensity + branchPerMem > 0.15` |
| `control_dominated` | `computeOps / (branches + 1) < 5.0` |
| `complex_kernel` | ≥ 2 of: {loops, barriers, highBranching} (suppressed for multi-kernel PTX sum) |
| `multi_kernel_ptx` | `ptxFeatures.kernelCount > 1 && SASS not present` |
| `tensor_dominated` | `tensor_ops > 0 && computeToMemory > 4.0` |
| `streaming` (micro) | `!hasShared && !hasBarrier && !hasLoops && computeToMemory < 2.0` |
| `sync_efficiency` | `"efficient"` / `"moderate"` / `"inefficient"` (workPerBarrier thresholds) |
| `interleaving` | `"interleaved"` / `"stacked"` / `"mixed"` (stream_interleave_score & max_consecutive_loads) |

### Step 5 — Inefficiency signal flags

| Flag | Formula / Condition | SASS Required? |
|------|---------------------|----------------|
| `spill_risk` | `local_loads + local_stores > 0` | Yes (always false without SASS) |
| `spill_severity` | `(local_loads + local_stores) / total_instructions` | Yes (0 when SASS unavailable) |
| `uncoalesced_risk` | `ldg_32 / totalLdgTyped > 0.75` AND `totalLdgTyped > 4` AND `globalOps > 4` | Yes |
| `store_uncoalesced_risk` | `stg_32 / totalStgTyped > 0.75` AND `totalStgTyped > 4` AND `globalOps > 4` | Yes |
| `missing_tensor_cores` | `!tensor_ops && computeToMemory > 4.0 && computeOps > 16` | No (SASS refines) |
| `atomic_contention_risk` | `global_atomic_ops > 0 && globalOps > 0 && global_atomic_ops / globalOps > 0.05` — uses **global-only** atomics; ATOMS (shared-memory) is excluded to prevent false positives | Yes |
| `sfu_heavy` | `sfu_ops > 8 && sfu_ops / (computeOps + sfu_ops + 1) > 0.15` | Yes |
| `vectorization_score` | `(ldg_128×4 + ldg_64×2 + ldg_32×1) / (totalLdgTyped × 4)` — active range **[0.25, 1.0]**; returns **0** (inactive, not "fully scalar") when SASS unavailable or `< 4` typed loads exist | Yes |
| `over_synchronized` | `effectiveLoops > 0 && barriers ≥ 2 && barriers / effectiveLoops > 1.5`, where `effectiveLoops = ptx.loops || sass.back_edges` | No (uses PTX loops + SASS barriers, with SASS back-edges as a fully-unrolled-loop substitute). When PTX reports `loops = 0` because the compiler unrolled the visible loop, `sass.back_edges` (backward `BRA <addr>` jumps) acts as the per-iteration count instead — so the signal can still fire on unrolled kernels. |
| `fp16_scalar_risk` | `fp16_arith_ops > 8 && wmma_ops = 0 && fp16_arith_ops / (computeOps + 1) > 0.2` | Yes |
| `read_modify_write` | `storeToLoadRatio ∈ (0.5, 2.0) && computeToMemory < 1.0 && globalOps > 4` | No (SASS preferred) |

### Step 5b — Additional Derived Pattern Signals (Groups A/C)

| Signal | Formula / Condition | Notes |
|--------|---------------------|-------|
| `tensor_utilization_fraction` | `sass.tensor_ops / (sass.arithmetic_ops + sass.tensor_ops + 1)` | `number \| undefined` — continuous tensor-core adoption score (0–1) when SASS is present; **`undefined`** when SASS was not provided (the UI then renders `—` instead of a hard `0`). |
| `productive_instruction_fraction` | `(sass.arithmetic_ops + sass.tensor_ops + sass.global_loads + sass.global_stores) / sass.total_instructions` | `number \| undefined` — instruction-level useful-work share when SASS is present; **`undefined`** for PTX-only kernels (PTX has no `total_instructions` denominator). |
| `shared_reuse_per_barrier` | `shared_loads / (barriers + 1)` | Cooperative phase amortization quality |
| `warp_divergence_risk` | `(branches × loops) / (computeOps + 1) > 0.01` | Loop-aware divergence risk |
| `store_vectorization_score` | `(stg_128×4 + stg_64×2 + stg_32×1) / (totalStgTyped × 4)` | 0.25–1.0 active range; 0 when inactive |
| `fp_to_int_ratio` | `(arithmetic_ops - integer_ops) / (integer_ops + 1)` | FP-vs-address/integer mix indicator |

> Both `tensor_utilization_fraction` and `productive_instruction_fraction` are now strictly SASS-derived; reporting them as a hard `0` for PTX-only kernels (the legacy behaviour) was misleading because the formulas had a `0/1` form that always evaluated to `0` regardless of how much arithmetic the kernel actually did.

### Step 6 — Stall Reason Inference (Group E)

These four signals infer the most likely warp-stall category from instruction-level evidence alone (no hardware performance counters required).  Each also appears inside `pattern_micro`.

| Flag | Condition | Interpretation |
|------|-----------|---------------|
| `stall_memory_dependency` (E1) | `stream_max_consecutive_loads > 4 AND stream_interleave_score < 0.2` | Long back-to-back load chain — warp stalls waiting for each load response before issuing the next |
| `stall_memory_throttle` (E2) | `globalOps > 16 AND stream_interleave_score > 0.4 AND computeToMemory < 2.0` | Good load/compute interleaving but very low arithmetic intensity — L2/DRAM pipeline is perpetually full |
| `stall_local_memory` (E3) | `(local_loads + local_stores) / total_instructions > 0.03` | > 3 % of instructions are register-spill I/O — warp repeatedly blocked on ≥ 100-cycle local-memory reloads |
| `stall_sync` (E4) | `barriers ≥ 2 AND workPerBarrier < 30 AND hasShared` | Very little compute between barriers — block spends most of its time at `__syncthreads()` waits |

> **Note for E3:** `diagnoseKernel()` cross-checks `stall_local_memory` against occupancy (< 0.5) before setting `stall_profile.local_memory = true`.  The pattern-level flag is unconditional.

### Step 7 — Warp-Level Primitives (Group F)

| Flag | Condition | Interpretation |
|------|-----------|---------------|
| `uses_warp_shuffle` (F2) | `warp_shuffle_ops > 0` | Kernel communicates register values between warp lanes via SHFL — no shared memory needed |
| `uses_warp_vote` (F3) | `warp_vote_ops > 0` | Kernel uses warp-collective boolean operations (`VOTE`/`MATCH`) |
| `warp_reduction_pattern` (F4) | `warp_shuffle_ops > 0 AND sfu_ops = 0 AND barriers > 0` | Inferred warp-shuffle cooperative reduction — SHFL + sync with no transcendental work |

### Step 8 — Kernel Archetype (Group H)

A single `archetype: string | undefined` field on `PatternResult` names the high-level kernel shape when multiple signals converge.  Rules are evaluated in priority order; first match wins.  `undefined` when no archetype is confidently matched.

| Value | Condition | Typical origin |
|-------|-----------|---------------|
| `"histogram_scatter"` (H4) | `read_modify_write AND atomic_contention_risk AND computeToMemory < 1.0` | Scatter accumulation, sparse histogram |
| `"gemm"` (H1) | `class = "tiled" AND usesTensor AND sharedToGlobal > 2.0 AND computeToMemory > 8.0 AND sharedLoads / (barriers+1) ≥ 8` | Tensor-core GEMM / convolution |
| `"attention"` (H2a) | `class = "reduction" AND sfu_ops > 4 AND barriers ≥ 2 AND workPerBarrier < 50 AND usesTensor` | Transformer attention (softmax fused with matmul) |
| `"softmax"` (H2b) | same as H2a but `!usesTensor` | Standalone softmax |
| `"reduction"` (H3) | `class = "reduction" AND barriers ≥ 2 AND sharedOps > 0 AND computeToMemory < 4.0 AND !usesTensor` | Tree / warp reduction |
| `"activation"` (H5) | `class = "elementwise" AND sfu_ops > 0 AND !usesTensor` | Activation function (ReLU with expf/logf variants) |

---

## Part 7 — Confidence Summary

Confidence reflects data-source quality.  Higher = more trustworthy classification.

| Model | Base | SASS bonus | Strong-signal bonus | Penalty / override conditions |
|-------|------|-----------|--------------------|-------------------------------|
| Memory | 0.60 | +0.25 | +0.10 (low intensity or reuse), +0.10 (LD.CS present) | `globalMemOps=0 & SASS absent` → 0.30 override; `globalMemOps=0 & SASS present` → no override; `<5 ops` → −0.10/−0.20 |
| Pattern | 0.55 | +0.10 | — | — |
| Occupancy | 0.65 | — | +0.10 **per launch-sourced field** (`threads`, `shared`, `registers`); +0.05 (`ptx.maxnreg`) | Cap at 0.85 |

---

## Part 8 — Diagnosis Layer (`diagnoseKernel`)

**Entry point:** `diagnoseKernel(memory: MemoryAnalysis, occ: KernelAnalysis, pattern: PatternResult): DiagnosisResult`

Fuses all three models into a single cross-model bottleneck report.  Operates entirely from previously computed outputs — no new instruction parsing.

### `DiagnosisResult` Interface

| Field | Type | Description |
|-------|------|-------------|
| `primary_bottleneck` | `string` | Tag for the dominant bottleneck (see rule table below); `"none_detected"` when no rules fire |
| `secondary_bottlenecks` | `string[]` | Tags for additional active bottlenecks, ordered by rule priority |
| `stall_profile.memory_dependency` | `boolean` | `pattern.stall_memory_dependency` |
| `stall_profile.memory_throttle` | `boolean` | `pattern.stall_memory_throttle` |
| `stall_profile.local_memory` | `boolean` | `pattern.stall_local_memory AND occupancy < 0.5` (cross-check) |
| `stall_profile.sync` | `boolean` | `pattern.stall_sync` |
| `optimization_priority` | `string[]` | Concrete fix suggestions, primary first then secondary |
| `confidence` | `number` | `min(rulesfired / 3, 1.0)` — 1.0 when ≥ 3 independent signals agree |

### G2 — Inference Rules (evaluated in priority order)

All rules are evaluated; the first match becomes `primary_bottleneck`; all subsequent matches populate `secondary_bottlenecks`.

| Priority | Tag | Trigger Condition | Suggested Fix |
|----------|-----|-------------------|---------------|
| 1 | `register_spill` | `pattern.spill_risk AND occupancy < 0.4` | Reduce registers; add `__launch_bounds__` |
| 2 | `memory_latency` | `memory.class = "memory_bound" AND pattern.stall_memory_dependency` | Prefetch; software pipelining |
| 3 | `memory_bandwidth` | `memory.class = "memory_bound" AND pattern.stall_memory_throttle` | Vectorize loads to LDG.128; eliminate redundant traffic |
| 4 | `atomic_contention` | `pattern.atomic_contention_risk AND pattern.read_modify_write` | Privatize; warp-shuffle reduction before atomic |
| 5 | `sfu_throughput` | `pattern.sfu_heavy AND memory.class ≠ "memory_bound"` | Replace MUFU with polynomial approximation |
| 6 | `sync_overhead` | `pattern.stall_sync AND pattern.class = "tiled"` | Merge tile phases; reduce barriers per loop |
| 7 | `uncoalesced_access` | `pattern.uncoalesced_risk OR pattern.store_uncoalesced_risk` | Transpose; SoA layout |
| 8 | `occupancy` | `occ.limiting_factor = "registers" AND occupancy < 0.4` | Use `__launch_bounds__` |
| 9 | `compute_bound` | `memory.class = "compute_friendly" AND pattern.missing_tensor_cores` | Switch to Tensor Core API |

---

## Part 9 — `analysis_mode` Detection

`AnalyzerReport.analysis_mode` advertises whether the displayed analysis is grounded in the same architecture the kernel was actually compiled for.

| Value | Meaning | Trigger |
|-------|---------|---------|
| `"native"` | The selected GPU preset matches the architecture of the SASS dump being analysed. | `expected_cc_key === sass_cc_key` for at least one detected SASS target. |
| `"cross-arch-what-if"` | SASS is present, but its target architecture differs from the selected preset. Numbers are projected onto the preset's SM limits. | A SASS dump is present but no `sass_cc_key` matches `expected_cc_key`. |
| `"preset-only-what-if"` | No SASS was provided — analysis is entirely PTX + preset-driven. | `sass_detected_targets` is empty. |

### Compute-capability key normalisation

Native detection compares **compute-capability keys**, not literal `sm_NN` numbers. The mapping in `SM_VERSION_TO_CC` collapses different SM numbers that share the same architecture-level CC into one key — so the `"native"` decision matches the way `GPU_SM_CONFIGS` defines architecture limits.

| `sm_NN` | `cc` key | Notes |
|---------|----------|-------|
| `sm_70` / `sm_72` | `7.0` | Volta |
| `sm_75` | `7.5` | Turing |
| `sm_80` | `8.0` | A100 |
| `sm_86` / `sm_87` | `8.6` | Ampere consumer / Jetson Orin (sm_87 → cc 8.6) |
| `sm_89` | `8.9` | Ada Lovelace |
| `sm_90` | `9.0` | Hopper |
| `sm_100` | `10.0` | Blackwell DC |
| `sm_120` | `12.0` | Blackwell consumer |

Before this normalisation, an `sm_87` (Jetson Orin) SASS dump compared against an `sm_86` (Ampere mobile) preset was reported as `cross-arch-what-if`, even though both architectures share identical `GPU_SM_CONFIGS` limits. The cc-key comparison fixes that mismatch and only flags genuine architectural transitions (e.g. Ampere → Hopper).

---

## Quick-Reference: Which Model Uses Which Raw Fields

| Raw Field | Memory | Occupancy | Pattern |
|-----------|:------:|:---------:|:-------:|
| `ptx.global_loads` | ✓ (fallback) | — | ✓ (fallback) |
| `ptx.global_stores` | ✓ (fallback) | — | ✓ (fallback) |
| `ptx.fma / add / mul` | ✓ (fallback flops) | — | ✓ (fallback compute) |
| `ptx.barrier` | — | — | ✓ (fallback) |
| `ptx.branches` | — | — | ✓ (fallback) |
| `ptx.loops` | — | — | ✓ (always) |
| `ptx.reg_decl_lines` | — | fallback register hint | — |
| `ptx.maxntid` (hint) | — | ✓ | — |
| `ptx.maxnreg` (hint) | — | ✓ | — |
| `ptx.static_shared` (hint) | — | ✓ | — |
| `sass.global_loads/stores` | ✓ | — | ✓ |
| `sass.ldg_128/64/32/16/8` | ✓ (byte calc) | — | ✓ (vectorization_score uses ≥32-bit buckets) |
| `sass.stg_128/64/32/16/8` | ✓ (byte calc + store_vectorization_score) | — | ✓ (store_vectorization_score, store_uncoalesced_risk) |
| `sass.cg_loads / cs_loads` | ✓ (cache_policy) | — | — |
| `sass.shared_loads/stores` | ✓ | — | ✓ |
| `sass.arithmetic_ops` | ✓ (scalar flops) | — | ✓ |
| `sass.integer_ops` | — | — | ✓ (`fp_to_int_ratio`) |
| `sass.tensor_ops` | ✓ (int MMA flops) | — | ✓ |
| `sass.wmma_ops` | ✓ (FP MMA flops) | — | ✓ |
| `sass.sfu_ops` | ✓ (SFU flops) | — | ✓ |
| `sass.fp16_arith_ops` | — | — | ✓ |
| `sass.fp64_arith_ops` | ✓ (FLOP-proxy weighting) | — | ✓ |
| `sass.const_loads` | — | — | (informational) |
| `sass.async_global_loads` | — | — | ✓ |
| `sass.tma_ops` | — | — | ✓ |
| `sass.atomic_ops` | — | — | ✓ (total serialization count) |
| `sass.global_atomic_ops` | — | — | ✓ (`atomic_contention_risk`) |
| `sass.local_loads/stores` | — | — | ✓ |
| `sass.barrier` | — | — | ✓ (Hopper/Blackwell barriers included) |
| `sass.branch` | — | — | ✓ (RET/EXIT excluded — see `kernel_exit`) |
| `sass.kernel_exit` | — | — | (informational) |
| `sass.back_edges` | — | — | ✓ (`over_synchronized` fallback when `ptx.loops = 0`) |
| `sass.max_register_index` | — | ✓ | — |
| `sass.stream_interleave_score` | — | — | ✓ (interleaving) |
| `sass.stream_max_consecutive_loads` | — | — | ✓ (interleaving) |
| `sass.warp_shuffle_ops` | — | — | ✓ (F2/F4 warp primitives) |
| `sass.warp_vote_ops` | — | — | ✓ (F3 warp primitives) |
