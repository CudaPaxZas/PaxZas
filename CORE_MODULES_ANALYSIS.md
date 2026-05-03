# Core Modules Analysis: PTX Kernel Synthesis Pipeline

## Overview

The Paxzas CUDA analyzer synthesizes GPU kernel analysis through three core interdependent modules that work together in a data flow pipeline:

1. **`parallel_kernel_extract.ts`** — Kernel Extraction & Parallelization Layer
2. **`ptx_features.ts`** — Instruction Feature Definition & Extraction Layer
3. **`memory_model.ts`** — Memory Analysis & Kernel Classification Layer

These modules are orchestrated by `analyze.ts`, which coordinates their execution and combines their outputs into a comprehensive **AnalyzerReport** that includes occupancy metrics, memory analysis, pattern classification, and bottleneck heuristics.

---

## Module 1: `parallel_kernel_extract.ts` — Kernel Extraction & Parallelization

### Purpose
Efficiently extract and aggregate instruction features from multiple GPU kernels in a single PTX file using parallel processing.

### Key Responsibilities

#### Kernel Discovery & Extraction
- Identifies all `.entry` kernel definitions in PTX code via `listPtxEntryNames()`
- Supports optional kernel filtering via substring matching (e.g., `kernelFilter="matmul"`)
- Locates kernel body spans (from opening `{` to closing `}`)

#### Parallel Processing
- Uses a **4-worker thread pool** (`KERNEL_EXTRACT_POOL_SIZE = 4`) to extract features from multiple kernels concurrently
- Falls back to sequential extraction if worker threads fail
- Implements custom `mapPool()` for controlled concurrency

#### Feature Aggregation
- **Single kernel:** Returns features for that kernel only
- **Multiple kernels:** Sums instruction features across all kernels using `sumPtxInstructionFeatures()`
- This aggregated feature set feeds downstream analysis (memory, pattern, occupancy models)

### Data Flow
```
PTX Text Input
    ↓
Parse Entry Names → Kernel Filter (optional)
    ↓
[Worker Pool] Extract Features per Kernel
    ↓
Sum Instruction Features (multi-kernel cases)
    ↓
PtxInstructionFeatures Output + Kernel Names List
```

### Return Structure
```typescript
{
  instr: PtxInstructionFeatures,      // Aggregated instruction counts
  kernelNames: string[]               // Analyzed kernel identifiers
}
```

### Integration Point
Returns structured data consumed by:
- `occupancy_model.ts` (picks first/filtered kernel for launch configuration)
- `memory_model.ts` (uses summed features for memory intensity analysis)
- `pattern_model.ts` (classifies based on aggregated instruction patterns)
- `bottleneck.ts` (heuristic analysis of bottleneck type)

---

## Module 2: `ptx_features.ts` — Instruction Feature Definition & Extraction

### Purpose
Define and extract quantifiable instruction metrics from PTX assembly code to enable data-driven analysis of kernel behavior.

### Core Data Structure: `PtxInstructionFeatures`
Captures high-level compute and memory characteristics:

```typescript
interface PtxInstructionFeatures {
  global_loads: number;      // ld.global instructions (memory reads)
  global_stores: number;     // st.global instructions (memory writes)
  fma: number;               // Fused multiply-add operations
  add: number;               // Scalar additions
  mul: number;               // Scalar multiplications
  barrier: number;           // Synchronization barriers (bar.sync)
  reg_decl_lines: number;    // Register declaration density
  branches: number;          // Branch operations
  loops: number;             // Loop constructs detected
}
```

### Feature Extraction Strategy

#### Instruction Pattern Matching
Uses fine-tuned regex patterns for robust PTX parsing:
- `LD_GLOBAL` → `ld.global` instructions (bandwidth-critical)
- `ST_GLOBAL` → `st.global` instructions (bandwidth-critical)
- `FMA` → Fused multiply-add (compute-intensive)
- `BARRIER` → `bar.sync` (synchronization overhead)
- `BRA_OPCODE` → Branch targets (control flow complexity)

#### Register & Loop Analysis
- Counts `.reg` declaration lines (indicates spill pressure)
- Detects labels and backward branches to infer loop presence
- Provides register pressure estimation

### Key Functions

#### `extractOneKernelFeatures(ptx, kernelSubstring)`
Single-kernel extraction: locates kernel body and counts all instruction types in that scope.

#### `extractInstructionFeaturesFromRange(ptx, start, end)`
Extracts features from a specific line range (used for per-kernel extraction in worker threads).

#### `sumPtxInstructionFeatures(perKernelArray)`
Aggregates features when multiple kernels exist:
```
aggregate.global_loads = kernel1.global_loads + kernel2.global_loads + ...
aggregate.fma = kernel1.fma + kernel2.fma + ...
// ... (all fields summed)
```

### Output Conversion
- `ptxFeaturesAsDict()` converts to dictionary for JSON serialization in analysis reports

### Integration Point
- Consumed by `memory_model.ts` for arithmetic intensity calculation
- Consumed by `pattern_model.ts` for behavior classification
- Consumed by `bottleneck.ts` for bottleneck type inference

---

## Module 3: `memory_model.ts` — Memory Analysis & Kernel Classification

### Purpose
Analyze kernel memory behavior to classify kernels, quantify memory pressure, and identify optimization strategies.

### Core Analysis: `MemoryAnalysis` Result

#### Instruction Counts (Instruction Sources)
- **Global Memory:** `global_loads`, `global_stores` (main bandwidth consumer)
- **Shared Memory:** `shared_loads`, `shared_stores` (fast local cache)
- **Aggregate:** `global_mem_ops`, `shared_mem_ops`

#### Memory Pressure Metrics
- **`memory_pressure`:** Total estimated bytes transferred
  - Combines SASS-level byte-width info (ldg_128, ldg_64, ldg_32) when available
  - Falls back to PTX-based heuristics (assume 4 bytes per unknown load/store)
  
#### Data Source Attribution
Tracks data provenance for confidence scoring:
- `global_mem_source` → "sass" or "ptx"
- `shared_mem_source` → "sass" or "unknown"
- `compute_source` → "sass" or "ptx"

#### Arithmetic Intensity (Key Metric)
```
Arithmetic Intensity = FLOPs / Bytes Transferred
```
- **High intensity** (> 10) → compute-friendly, not memory-bound
- **Low intensity** (< 1) → memory-bound, optimize memory access patterns
- **Medium intensity** (1–10) → balanced, context-dependent

#### Cache & Memory Behavior
- **`cache_policy`** → Detected cache level ("L2", "streaming", null)
- **`confidence`** → Reliability score (0.0–1.0) based on data sources

### Kernel Classification

Memory model logically classifies kernels:

| Classification     | Condition | Optimization Target |
|--------------------|-----------|---------------------|
| `compute_friendly` | High compute/memory ratio | Throughput (not bandwidth-limited) |
| `memory_bound`     | intensity < 0.5 && globalMemOps ≥ 4 (Gap 3) | Memory access patterns |
| `reuse_optimized`  | Effective shared memory usage | Cache hierarchy & data reuse |
| `balanced`         | No strong imbalance | General optimizations |

**Gap 3 fix:** the old `bytesMoved > 2048` guard excluded small-but-clearly-memory-bound
kernels (e.g. 24 LDG.32 = 96 bytes at intensity 0.03).  Replaced by `globalMemOps >= 4`
which correctly classifies any kernel with meaningful bandwidth demand.

**Confidence scoring improvements:**
- Gap 5: `isStreaming` (LD.CS) adds +0.1 confidence (unambiguous signal)
- Gap 6: `globalMemOps < 5` subtracts 0.1–0.2 (too few data points for reliable classification)

### Key Functions

#### `estimateSassBytes(sass, defaultBytesPerOp)`
When SASS data is available, calculates precise byte counts:
```
bytes = ldg_128 × 16 + ldg_64 × 8 + ldg_32 × 4
      + (unknown_loads  × default)
      + stg_128 × 16 + stg_64 × 8 + stg_32 × 4   ← Gap 4: typed store widths
      + (unknown_stores × default)
```
Before Gap 4 all stores were assumed 4 bytes, causing a 4× underestimate of write
bandwidth for kernels using STG.E.128 (coalesced 128-bit stores).

#### Compute Intensity Calculation
```
// Gap 1: Fix tensor FLOP proxy (was 2×; now 512× for FP MMA, 64× for int MMA)
// Gap 2: Include SFU ops (MUFU ≈ 4 float-op equivalents)
scalarFlops  = arithmetic_ops × 2
tensorFlops  = wmma_ops × 512 + (tensor_ops − wmma_ops) × 64
sfuFlops     = sfu_ops × 4
flops = scalarFlops + tensorFlops + sfuFlops
bytes = estimateSassBytes(sass) OR ptxBytesHeuristic()
intensity = flops / bytes  (safe division: 0 if bytes == 0)
```

### Integration Point
- Consumes `PtxInstructionFeatures` from `parallel_kernel_extract.ts`
- Works alongside SASS features (when supplemental SASS is provided)
- Output fed to `analyze.ts` → `AnalyzerReport.memory`
- Used in conjunction with occupancy and pattern analysis for comprehensive kernel characterization

---

## Module 0 (Optional): `sass_features.ts` — Low-Level Assembly Feature Extraction

### Purpose
Extract precise instruction-level metrics from SASS (low-level GPU assembly) to enhance analysis accuracy with exact byte-width information and advanced instruction classifications.

**SASS vs PTX:** While PTX is high-level intermediate representation estimated via heuristics, SASS is the actual compiled GPU assembly code executed on hardware, providing ground truth for instruction counts and memory access patterns.

### Why SASS Matters

| Metric | PTX (Heuristic) | SASS (Actual) | Impact |
|--------|-----------------|---------------|--------|
| Byte widths per load | Assumed 4B | Exact (ldg_32, ldg_64, ldg_128) | ±300% accuracy difference |
| Total instructions | Count all ops | Actual hardware instructions | Occupancy calculation |
| Cache policy | Inferred | Explicit (.CG, .CS modifiers) | L2 vs streaming behavior |
| Tensor operations | Not detected | Direct count (MMA, HMMA, WGMMA) | Critical for modern GPUs |
| Register usage | Declared `.reg` lines | Actual max register index | Spill detection |

### Core Data Structure: `SassInstructionFeatures`

```typescript
interface SassInstructionFeatures {
  // Global Memory (Device RAM)
  global_loads: number;           // Total ld.global instructions
  global_stores: number;          // Total st.global instructions
  
  // Byte-Width Precision for loads (SASS-specific)
  ldg_128: number;                // 128-bit loads (16 bytes)
  ldg_64: number;                 // 64-bit loads (8 bytes)
  ldg_32: number;                 // 32-bit loads (4 bytes)

  // Byte-Width Precision for stores — Gap 4 addition
  stg_128: number;                // 128-bit stores (16 bytes)
  stg_64: number;                 // 64-bit stores (8 bytes)
  stg_32: number;                 // 32-bit stores (4 bytes)
  
  // Cache Behavior (SASS-specific)
  cg_loads: number;               // Coherent Global (L2 cached)
  cs_loads: number;               // Coherent Streaming (bypasses L1)
  
  // Shared Memory (Fast local cache)
  shared_loads: number;           // lds (load shared)
  shared_stores: number;          // sts (store shared)
  
  // Compute Operations (Precise classification)
  arithmetic_ops: number;         // FFMA, FADD, FMUL, IADD, IMAD, etc.
  tensor_ops: number;             // ALL tensor ops: MMA, HMMA, IMMA, BMMA, WGMMA, WMMA
  /** FP tensor ops only: HMMA, WGMMA, WMMA — excludes integer (IMMA) and binary (BMMA). */
  wmma_ops: number;
  
  // Synchronization & Control Flow
  barrier: number;                // BAR, DEPBAR, MEMBAR
  branch: number;                 // BRA, JMP, RET, SSY, SYNC (incl. predicated @P0 BRA)
  
  // Register Spill Detection (Local Memory)
  /** LDL: restores a value the compiler evicted from the register file to local memory. */
  local_loads: number;
  /** STL: writes a live value into local memory to free up a physical register. */
  local_stores: number;

  // Serialisation & Special-Purpose Counters (SASS-only)
  /** ATOM/ATOMS/RED: global and shared-memory atomic ops — all serialise at L2 or shared bank. */
  atomic_ops: number;
  /** MUFU.*: Special Function Unit instructions (SIN, COS, EXP2, LOG2, RCP, RSQ, SQRT) — ¼ ALU throughput. */
  sfu_ops: number;
  /** HFMA/HADD/HMUL: scalar FP16 CUDA-core ops; subset of arithmetic_ops (NOT tensor-core). */
  fp16_arith_ops: number;

  // Aggregate Metrics
  total_instructions: number;     // Complete kernel instruction count
  max_register_index: number;     // Highest register used (infers reg count)
  
  // Streaming Metrics (Pattern analysis)
  instruction_sequence: string[]; // Ordered opcode list for pattern matching
  stream_interleave_score: number; // Measure of load/compute interleaving
  stream_max_consecutive_loads: number; // Stall risk indicator
}
```

### SASS Feature Extraction: `extractSassFeatures()`

#### Single-Pass Scan Strategy
- **Function discovery:** Searches for `^ *Function\s*:\s*(.+)` line (first or filtered match)
- **Opcode extraction:** Regex strips predicate guards (`@P0`, `@!P1`) then captures the opcode — without this, warp-divergence instructions like `@P0 BRA 0x200` were silently dropped
- **Register tracking:** Updates `max_register_index` when `R<N>` register references appear
- **Bounded processing:** Stops at next `Function:` (only analyzes first or matched kernel)

#### Opcode Classification
Each SASS opcode is categorised (predicate prefix stripped first):

```
// Strip optional predicate guard before matching:
//   @P0 BRA 0x100  →  opcode = "BRA"
//   @!P1 BRA 0x200 →  opcode = "BRA"

IF opcode.startsWith("LDG"):
  global_loads++
  IF ".128" in opcode: ldg_128++
  ELSE IF ".64"/.U64"/.F64" in opcode: ldg_64++
  ELSE IF ".32"/.U32"/.F32" in opcode: ldg_32++
  IF ".CG" in opcode: cg_loads++        // Cache policy
  IF ".CS" in opcode: cs_loads++

IF opcode.startsWith("STG"):  // Gap 4: typed store widths
  global_stores++
  IF ".128" in opcode: stg_128++
  ELSE IF ".64"/.U64"/.F64" in opcode: stg_64++
  ELSE IF ".32"/.U32"/.F32" in opcode: stg_32++
IF opcode.startsWith("LDS"):  shared_loads++
IF opcode.startsWith("STS"):  shared_stores++

// Register spill detection:
// LDL/STL prove the compiler exhausted the register file and had to evict
// live values to per-thread local memory (L1/L2/DRAM backed). Even a few
// LDL per warp thread cause 100+ cycle stalls.
IF opcode.startsWith("LDL"):  local_loads++
IF opcode.startsWith("STL"):  local_stores++

IF opcode IN [FFMA, FADD, FMUL, IADD, IMAD, ...]:
  arithmetic_ops++
  IF opcode IN [HFMA, HADD, HMUL]:
    fp16_arith_ops++          // Scalar FP16 subset — subset of arithmetic_ops

IF opcode IN [MMA, HMMA, IMMA, BMMA, WGMMA, WMMA]:
  tensor_ops++              // All tensor-core operations
  IF opcode IN [HMMA, WGMMA, WMMA]:
    wmma_ops++              // FP tensor only (excludes IMMA, BMMA)

IF opcode IN [ATOM, ATOMS, RED]:
  atomic_ops++              // Global + shared-memory atomics

IF opcode.startsWith("MUFU"):
  sfu_ops++                 // Special Function Unit (transcendentals)

IF opcode IN [BRA, JMP, RET, SSY, SYNC]:  branch++
```

**Why separate `wmma_ops`?** `IMMA` (integer matrix accumulate) and `BMMA` (binary) are tensor ops but do not help floating-point throughput. `wmma_ops` isolates HMMA/WGMMA/WMMA — the FP matrix units — so the pattern model can ask "does this GEMM-like kernel actually use FP tensor cores?" independently of integer tensor traffic.

#### Streaming Metrics
Tracks instruction sequence for pattern analysis:
- **Interleave Score:** Measures load/store/compute interleaving (high = good overlap potential)
- **Max Consecutive Loads:** Detects load stall chains (high = memory stall risk)

### Key Functions

#### `extractSassFeatures(sassText, kernelSubstring)`
Returns `[kernelName, SassInstructionFeatures]`:
```typescript
const [sassKernel, sassInstr] = extractSassFeatures(sassText, "matmul");
// sassKernel = "matmul" (matched name)
// sassInstr = { global_loads: 42, ldg_128: 10, ldg_64: 5, ... }
```

#### `inferRegistersPerThreadFromSass(sassText, kernelSubstring)`
Extracts exact register count from SASS:
```typescript
const regsFromSass = Math.max(0, max_register_index + 1);  // e.g., R255 → 256 regs
```

#### `buildSassFeatureBundle(registersPerThread, instr)`
Combines register count with features for reporting:
```typescript
{
  global_loads: 42,
  ldg_128: 10,
  ldg_64: 5,
  registers: 256,
  inferred_registers_per_thread: 256
}
```

### Integration with Analysis Pipeline

#### Within `analyze.ts` Flow
```
1. Check if file contains SASS (looksLikeSassDump() regex check)
2. Extract SASS features in parallel with PTX parsing
3. Infer register count from max_register_index
4. Use SASS register count as override for PTX .maxnreg declaration
5. Pass sassInstr to runModelsParallelOrSync()
```

#### Downstream Consumption

**Memory Model Enhancement:**
```typescript
// PTX only: Assume 4 bytes per load/store
memory_pressure = global_loads * 4 + global_stores * 4

// SASS + PTX: Precise byte widths for loads AND stores (Gap 4)
memory_pressure = ldg_128 * 16 + ldg_64 * 8 + ldg_32 * 4
                + unknown_loads * 4
                + stg_128 * 16 + stg_64 * 8 + stg_32 * 4
                + unknown_stores * 4

// FLOP proxy (Gaps 1 + 2)
flops = arithmetic_ops * 2              // scalar ops
      + wmma_ops * 512                  // FP tensor-core (was: 2×)
      + (tensor_ops - wmma_ops) * 64   // integer/binary MMA
      + sfu_ops * 4                     // MUFU transcendentals (was: missing)
**Pattern Model:**
- Uses `instruction_sequence` for exact opcode sequence analysis
- Leverages `stream_interleave_score` and `stream_max_consecutive_loads`
- Detects tensor core usage via `tensor_ops` / `wmma_ops`
- `local_loads + local_stores > 0` → `spill_risk = true` in `PatternResult`
- `ldg_32 / total_typed_ldg > 0.75` → `uncoalesced_risk = true`
- `atomic_ops / globalOps > 0.05` → `atomic_contention_risk = true`
- `sfu_ops > 8 && sfu_ops/(computeOps+sfu_ops) > 0.15` → `sfu_heavy = true`
- `fp16_arith_ops > 8 && wmma_ops === 0 && fp16/computeOps > 0.2` → `fp16_scalar_risk = true`
- `(ldg_128×4 + ldg_64×2 + ldg_32×1) / (totalTyped×4)` → continuous `vectorization_score`

**Occupancy Model:**
- Replaces PTX register estimate with exact `inferred_registers_per_thread`
- More accurate occupancy calculation (critical bottleneck in register-starved kernels)

### SASS-Only Analysis Path

When SASS file is provided **without PTX**:

```
SASS Input (no PTX)
    ↓
extractSassFeatures() → SassInstructionFeatures + max_register_index
    ↓
Infer: threads=256 (default), shared=0 (default), regs=max_register_index+1
    ↓
Memory + Occupancy + Pattern + Bottleneck models
    ↓
AnalyzerReport (kind: "sass_only", note: "PTX-free analysis with assumed launch params")
```

Example output:
```typescript
{
  kind: "sass_only",
  ptx_kernel: "matmul",          // SASS function name
  ptx_hints: null,               // No PTX metadata
  ptx_features: { ... },         // Built from SASS instruction counts
  register_source: "sass.inferred",  // Ground truth from assembly
  register_estimates: {
    ptx_maxnreg: undefined,
    sass_inferred: 128,           // Exact from SASS
    selected_registers_per_thread: 128
  },
  sass_note: "SASS-only file: occupancy uses threads=256, shared=0..."
}
```

---

## Synthesis: How The Modules Work Together

### Flow Diagram: From Raw PTX & SASS to Comprehensive Analysis

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: PTX Code + Optional SASS + Optional Launch Parameters │
└────────────────────┬─────────────────────────────────────────┘
                     │
        ┌────────────┴──────────────┐
        │    SASS EXTRACTION        │ (Optional: if SASS file provided)
        │    (sass_features.ts)     │
        └────────────┬──────────────┘
                     │
        ┌────────────────────────────────────────┐
        │         MERGE LAUNCH CONFIG            │
        │    (threads/shared/regs + SASS hints)  │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────────────────────────────────┐
        │ PARALLEL_KERNEL_EXTRACT                │
        │ ├─ Discover .entry kernels            │
        │ ├─ Extract per-kernel features (4x)   │
        │ └─ Aggregate PTX instruction counts   │
        └────────────┬──────────────────────────┘
                     │
         ┌───────────┴────────────────────────────┐
         │                                        │
    ┌────▼──────────────┐              ┌─────────▼────────┐
    │ PtxInstructionFeatures          │ SassInstructionFeatures
    │ ├─ global_loads                 │ ├─ global_loads (precise)
    │ ├─ global_stores                │ ├─ ldg_128/64/32
    │ ├─ fma, add, mul                │ ├─ shared_loads/stores
    │ ├─ barriers                     │ ├─ arithmetic_ops
    │ └─ loops, branches              │ ├─ tensor_ops (MMA)
    └────┬──────────────┘              │ ├─ cache_policy
         │                              │ └─ instruction_sequence
         │                              └─────────┬────────┘
         │                                        │
         └────────────┬─────────────────────────┬─┘
                      │ (merged/overlayed)      │
        ┌─────────────────────────────────────────────┐
        │  FEATURE FUSION LAYER                       │
        │  (memory_model uses both PTX + SASS)        │
        │  Selects best data source per metric:       │
        │  - SASS byte-widths override PTX heuristics │
        │  - SASS cache policy replaces inference     │
        │  - PTX loop structure for pattern fills gap │
        └──────────────┬──────────────────────────────┘
                       │
        ┌──────────────┴─────────────────┬──────────────────┬────────────────┐
        │                                │                  │                │
   ┌────▼──────┐            ┌──────────▼────────┐  ┌─────▼────────┐  ┌───▼──────────┐
   │ OCCUPANCY  │            │  MEMORY_MODEL     │  │ PATTERN_MODEL │  │ BOTTLENECK   │
   │ _MODEL     │            │ ├─ Compute I.O.   │  │ ├─ Ops seq    │  │ ├─ Heuristic │
   │ ├─ Threads │            │ ├─ Memory Pressure│  │ ├─ Interleave │  │ └─ Type      │
   │ ├─ Shared  │            │ ├─ Classification │  │ ├─ Cache use  │  └──────────────┘
   │ ├─ Regs*   │            │ └─ Confidence     │  │ └─ Compute%  │
   │ └─ Occ %   │            └───────────────────┘  └───────────────┘
   └────────────┘
    (* uses SASS
    inferred regs
    if available)
        │                          │                  │
        │                          │                  │
        └──────────────┬───────────┴──────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   ANALYZE.TS ORCHESTRATION   │
        │   ├─ Combine all results    │
        │   ├─ Cross-validate metrics │
        │   ├─ Track data sources     │
        │   └─ Synthesize report      │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │    AnalyzerReport                │
        │    ├─ kind: "ptx" | "sass_only" │
        │    ├─ occupancy_model            │
        │    ├─ memory                     │
        │    ├─ pattern                    │
        │    ├─ bottleneck_heuristic       │
        │    ├─ ptx_features + sass notes  │
        │    ├─ register_source tracking   │
        │    └─ gpu_spec_resolution        │
        └─────────────────────────────────┘
```

### Concrete Example: Matrix Multiply Kernel (PTX vs SASS)

**PTX Assembly:**
```ptx
.entry matmul(
  .param .u64 a,
  .param .u64 b,
  .param .u64 c
)
{
  ld.global.f32  %f0, [%rd0];     // global_loads++
  ld.global.f32  %f1, [%rd1];     // global_loads++
  fma.rn.f32     %f2, %f0, %f1, %f2;  // fma++
  st.global.f32  [%rd2], %f2;     // global_stores++
  bar.sync 0;                     // barrier++
}
```

**Compiled SASS Assembly (Ampere):**
```sass
Function : matmul
/*0000*/   LDG.E.SYS.128 R0:4, [R0.X] ;     // 128-bit load = 16 bytes
/*0010*/   LDG.E.SYS.64  R4:2, [R8.X] ;     // 64-bit load = 8 bytes
/*0020*/   FFMA.RN R12, R0, R4, R12 ;       // Floating multiply-add
/*0030*/   STG.E.SYS     [R16.X], R12 ;     // Global store
/*0040*/   BAR.SYNC 0 ;                     // Barrier sync
```

#### Step 1: Extraction (parallel_kernel_extract.ts + sass_features.ts)

**PTX Analysis:**
```
Kernel found: "matmul"
╔═══════════════════════════════╗
║ PtxInstructionFeatures:       ║
║ ├─ global_loads: 2            ║
║ ├─ global_stores: 1           ║
║ ├─ fma: 1                     ║
║ ├─ add: 0                     ║
║ └─ barrier: 1                 ║
╚═══════════════════════════════╝
```

**SASS Analysis (same kernel):**
```
Kernel found: "matmul"
╔═════════════════════════════════════┓
║ SassInstructionFeatures:            ║
║ ├─ global_loads: 2 (precise!)       ║
║ ├─ ldg_128: 1 (16 bytes exact)      ║
║ ├─ ldg_64: 1 (8 bytes exact)        ║
║ ├─ global_stores: 1                 ║
║ ├─ arithmetic_ops: 1 (FFMA)         ║
║ ├─ barrier: 1                       ║
║ ├─ max_register_index: 15 → 16 regs ║
║ └─ instruction_sequence: [...] 5 op │
╚═════════════════════════════════════╝
```

#### Step 2: Memory Analysis with Data Fusion (memory_model.ts)

**PTX-Only Path (Heuristic):**
```
PtxInstructionFeatures → Memory Model
├─ Compute: 1 FLOP per cycle (fma only)
├─ Memory Ops: 3 (2 loads + 1 store)
├─ Memory Pressure: 3 × 4 bytes = 12 bytes (assumed)
├─ Arithmetic Intensity: 1 FLOP / 12 bytes = 0.083
├─ Classification: MEMORY_BOUND
└─ Data source: "ptx" (confidence: medium)
```

**PTX + SASS Path (Exact):**
```
Fusion: Use SASS byte-widths, PTX instruction counts
├─ Compute: 1 FLOP (from PTX FFMA)
├─ Memory Ops: 3 (2 loads + 1 store)
├─ Memory Pressure: ldg_128(16) + ldg_64(8) + store(4) = 28 bytes
├─ Arithmetic Intensity: 1 FLOP / 28 bytes = 0.036
├─ Cache policy: .SYS (from SASS opcodes) → streaming bypasses L1
├─ Classification: MEMORY_BOUND (stronger signal)
├─ Register usage: 16 (exact from SASS, not estimated)
└─ Data source: "sass" (confidence: high)
```

**Key Insight:** SASS shows actual 128+64-bit loads (wider than PTX hints), resulting in 3× higher memory pressure estimate and stronger memory-bound classification.

#### Step 3: Synthesis (analyze.ts)

**PTX-Only Report:**
```typescript
{
  kind: "ptx",
  ptx_kernel: "matmul",
  ptx_features: { global_loads: 2, global_stores: 1, fma: 1, ... },
  register_source: "ptx",                       // Less precise
  register_estimates: {
    ptx_maxnreg: undefined,
    sass_inferred: null,
    selected_registers_per_thread: 32            // Default assumption
  },
  memory: {
    global_loads: 2,
    global_stores: 1,
    memory_pressure: 12,
    arithmetic_intensity: 0.083,
    classification: "memory_bound",
    global_mem_source: "ptx",
    confidence: 0.65                             // Medium confidence
  },
  occupancy_model: { threads: 256, occupancy_pct: 75 }
}
```

**PTX + SASS Report:**
```typescript
{
  kind: "ptx",
  ptx_kernel: "matmul",
  ptx_features: { global_loads: 2, global_stores: 1, fma: 1, ... },
  register_source: "sass.inferred",              // Ground truth
  register_estimates: {
    ptx_maxnreg: undefined,
    sass_inferred: 16,
    selected_registers_per_thread: 16            // Exact from SASS
  },
  memory: {
    global_loads: 2,
    global_stores: 1,
    ldg_128: 1,                                  // SASS-specific precision
    ldg_64: 1,
    memory_pressure: 28,                         // 3× more accurate
    arithmetic_intensity: 0.036,
    classification: "memory_bound",
    cache_policy: "streaming",                   // From SASS .SYS
    global_mem_source: "sass",                   // Prefers SASS data
    confidence: 0.92                             // High confidence
  },
  occupancy_model: { threads: 256, occupancy_pct: 98 },  // Better with exact regs
  sass_note: null
}
```

---

## Key Design Principles

### 1. **Separation of Concerns**
- **Extraction** (parallel_kernel_extract) → isolated feature discovery
- **Definition** (ptx_features) → reusable instruction metrics
- **Analysis** (memory_model) → semantic interpretation of metrics

### 2. **Parallelization for Scale**
- Worker threads extract per-kernel features concurrently
- Graceful fallback to sequential extraction if workers fail
- Enables near-linear scaling on multi-core systems

### 3. **Data Provenance Tracking**
- All outputs track source (PTX vs SASS) for confidence scoring
- Enables cross-validation when both PTX and SASS are available
- Informs users about analysis reliability

### 4. **Composability**
- Each module produces structured, JSON-serializable output
- Outputs feed cleanly to downstream analysis tasks
- Easy to extend or replace analysis stages

### 5. **Heuristic Robustness**
- Regex-based parsing resilient to formatting variations
- Safe division operations (0 on division by zero)
- Optional fields handle incomplete data gracefully

---

## Extensions & Future Integration Points

### Pattern Model Integration
- Consumes aggregated `PtxInstructionFeatures` + optional `SassInstructionFeatures`
- Compares instruction ratios against learned patterns
- Returns behavior classification (compute-intensive, memory-bound, etc.)
- **Inefficiency signals** surfaced on every `PatternResult`:

| Field | Signal | Detection Method |
|-------|--------|------------------|
| `spill_risk` | Register file exhausted → LDL/STL traffic | `local_loads + local_stores > 0` (SASS) |
| `uncoalesced_risk` | Non-contiguous warp access → serialised sub-transactions | `ldg_32 / typed_ldg > 75%` AND `global_ops > 4` (SASS) |
| `missing_tensor_cores` | FFMA-heavy GEMM with no MMA/HMMA | `!tensor_ops && compute_to_memory > 4 && computeOps > 16` |
| `uses_tensor_cores` | Tensor core instructions present | `sassFeatures.tensor_ops > 0` |
| `atomic_contention_risk` | ATOM/RED serialisation dominates global traffic | `atomic_ops / globalOps > 5%` (SASS) |
| `sfu_heavy` | Special Function Unit bottleneck (¼ ALU throughput) | `sfu_ops > 8 && sfu_ops / (computeOps + sfu_ops) > 15%` (SASS) |
| `vectorization_score` | Weighted average load width (1.0 = all LDG.128, 0.25 = all LDG.32) | `(ldg_128×4 + ldg_64×2 + ldg_32×1) / (totalTyped×4)` — 0 when < 4 typed loads |
| `over_synchronized` | Too many `__syncthreads()` per loop iteration | `barriers / loops > 1.5 && barriers ≥ 2 && loops > 0` (PTX + SASS) |
| `fp16_scalar_risk` | Heavy scalar FP16 (HFMA) without tensor-core path (HMMA) | `fp16_arith_ops > 8 && wmma_ops === 0 && fp16 / computeOps > 20%` (SASS) |
| `read_modify_write` | Load-then-store scatter / histogram pattern | `storeToLoadRatio ∈ (0.5, 2.0) && computeToMemory < 1.0 && globalOps > 4` |

All ten flags appear at the top level of `PatternResult` **and** mirrored inside `pattern_micro` for backward compatibility.

### Occupancy Model Integration
- Uses thread/shared/register configuration
- GPU spec (compute capability)
- Calculates theoretical occupancy % and throughput
- **Gap 7 fix:** registers allocated per-warp (`warpsPerBlock × roundUp(R×32, unit)`) not per-block — the old per-block rounding overstated occupancy for certain register counts
- **Gap 8:** `blocks_per_sm = 0` emits a hard launch-failure warning and skips all other diagnostics
- **Gap 9:** `KernelAnalysis.occupancy_class` field (`"low"` / `"medium"` / `"high"`) now exposed directly
- **Gap 10:** Limiting-factor warnings now include computed fix targets: registers show `≤N regs/thread` with `__launch_bounds__`; shared memory shows `≤N bytes/block`
- **Gap 11:** `KernelAnalysis.estimated_sm_utilization` surfaces device-level throughput: `"N / M SMs active"` when `smCount` is known

### Multi-GPU Analysis
- Memory model adjusts bandwidth assumptions by GPU architecture
- Occupancy model selects correct computation rules by CC
- Bottleneck heuristic calibrated per architecture

---

## Summary

The synthesis pipeline **transforms low-level code (PTX or SASS) into actionable GPU kernel insights:**

| Stage | Input | Process | Output | Data Source |
|-------|-------|---------|--------|-------------|
| **Extract (SASS)** | Raw SASS assembly | Opcode scan (predicate-stripped), byte-width parsing, LDL/STL spill detection, HMMA/wmma sub-count | `SassInstructionFeatures` | Hardware-compiled code |
| **Extract (PTX)** | Raw PTX | Find kernels, count instructions (parallel) | `PtxInstructionFeatures` | Intermediate representation |
| **Fuse** | Both feature sets | Select best source per metric (SASS precision > PTX heuristics) | Merged features | Hybrid optimisation |
| **Analyze (Memory)** | Merged features | Compute intensity, memory pressure, classify | `MemoryAnalysis` | Semantic interpretation |
| **Analyze (Pattern)** | Merged features | Pattern class + 10 inefficiency flags (spill, uncoalesced, missing TC, atomic contention, SFU bottleneck, vectorization score, over-sync, FP16 scalar risk, read-modify-write) | `PatternResult` | Semantic interpretation |
| **Synthesize** | Memory + Occupancy + Pattern + Bottleneck | Cross-validate, correlate, confidence scoring | `AnalyzerReport` | Comprehensive insights |

### SASS Integration Impact

When **SASS is available**, analysis accuracy improves dramatically:

| Metric | PTX-Only | SASS-Enhanced | Gain |
|--------|----------|---------------|----|
| Byte-width precision — loads | ±300% heuristic error | Exact counts (ldg_32/64/128) | 3-10× more accurate |
| Byte-width precision — stores | all assumed 4 B | Exact (stg_32/64/128) — Gap 4 | up to 4× more accurate |
| Register count | Estimated from .reg lines | Exact max_register_index | Near-perfect |
| Cache behavior | Inferred | Explicit (.CG, .CS modifiers) | 100% precise |
| Tensor FLOP proxy | 2× per instruction | 512× FP MMA, 64× int MMA — Gap 1 | ~256× more accurate |
| SFU FLOP proxy | Missing | 4 ops/MUFU — Gap 2 | SFU kernels no longer mis-classified |
| Instruction sequence | Unavailable | Complete opcode list | Pattern matching enabled |
| Tensor core usage | Not detected | `tensor_ops` (all MMA types) + `wmma_ops` (FP only) | FP vs INT tensor distinction |
| Warp divergence | Branch count only (PTX) | Predicated `@P0 BRA` correctly counted (SASS) | No under-counting |
| Register spill | Invisible | `local_loads` + `local_stores` (LDL/STL) | Direct spill evidence |
| Uncoalesced access | No signal | Narrow-load ratio (`ldg_32 / typed_ldg`) | Warp-serialisation risk |
| Confidence score | 0.65 (medium) | 0.92 (high) | 40% boost |

Together, these **three core modules + optional SASS** provide **comprehensive, parallelizable, and extensible analysis** of CUDA kernel performance characteristics within VS Code.

---

## Appendix: Practical SASS Usage Guide

### How to Obtain SASS Files

#### 1. **NVIDIA CUDA Toolkit (`cuobjdump`)**
```bash
# Compile to PTX
nvcc -ptx mykernel.cu -o mykernel.ptx

# Dump SASS for specific GPU architecture
cuobjdump -arch=sm_86 mykernel.ptx > mykernel.sass

# Or directly from compiled binary
cuobjdump -sass mykernel.cubin > mykernel.sass
```

#### 2. **NVIDIA Nsight Compute**
- Profiles kernels and generates detailed SASS dumps
- GUI-friendly inspection of instruction sequences
- Cache behavior and memory hierarchy analysis

#### 3. **NVDISASM (NVIDIA Disassembler)**
```bash
nvdisasm -c -hex mykernel.cubin > mykernel.sass
```

### Using SASS with Paxzas

#### Option A: Analyze PTX + SASS (Most Accurate)
```
Open mykernel.ptx in VS Code editor
→ Command: "Paxzas: Analyze with Launch Spec"
→ (Optional) Provide threads=128,shared=48,etc.
→ System auto-detects supplemental SASS if named `mykernel.sass` 
  in same directory (or use settings to specify path)
```

#### Option B: Analyze SASS Only
```
Open mykernel.sass in VS Code editor
→ Command: "Paxzas: Analyze Active CUDA File"
→ Report uses inferred defaults: threads=256, shared=0
→ Occupancy calculated with exact SASS register count
```

#### Option C: Manual SASS Merge
```typescript
// In extension code or custom script:
const ptxText = readFileSync('mykernel.ptx', 'utf-8');
const sassText = readFileSync('mykernel.sass', 'utf-8');
const report = await analyze(ptxText, {}, undefined, sassText);
```

### Interpreting SASS Output

#### Register Pressure
```output
register_source: "sass.inferred"          ← Ground truth
register_estimates: {
  sass_inferred: 128,                     ← Exact from SASS
  selected_registers_per_thread: 128
}
```
**Meaning:** Kernel uses exactly 128 registers/thread (no guessing).

#### Memory Access Pattern
```output
memory: {
  global_loads: 42,
  ldg_128: 10,      ← 10 wide (128-bit) loads
  ldg_64: 5,        ← 5 medium (64-bit) loads
  ldg_32: 27,       ← 27 narrow (32-bit) loads
  cache_policy: "streaming"   ← .CG (cached) or .CS (bypass L1)
}
```
**Meaning:** May have unaligned/irregular access patterns (27 narrow vs 10 wide loads suggest non-unit stride).

#### Instruction Sequence Analysis
```output
stream_interleave_score: 0.45     ← Low = poor compute/memory overlap
stream_max_consecutive_loads: 8   ← Stall risk (8 loads in a row)
```
**Meaning:** Consider adding compute operations or prefetching to hide memory latency.

### When SASS is Not Available

If you only have PTX (no compiled SASS):
- Paxzas still works (`kind: "ptx"`)
- Uses heuristic estimates for byte-widths (assume 4B per access)
- Confidence score is lower (~0.65 vs 0.92 with SASS)
- Register estimates based on `.maxnreg` declaration (may not reflect actual usage)
- Occupancy calculation less precise

**Recommendation:** Always collect SASS when possible for analysis accuracy.
