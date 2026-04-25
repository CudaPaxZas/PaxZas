/**
 * Pattern classification (matches pattern_model.py).
 *
 * This module classifies GPU kernels into computational patterns based on their
 * instruction characteristics. Pattern recognition helps identify optimization
 * strategies and expected bottlenecks.
 *
 * Kernel patterns and their characteristics:
 * 1. control_heavy: High branching/conditional logic relative to computation
 *    -> Problem: Branch divergence, reduced SIMD efficiency
 *    -> Fix: Restructure to reduce branch count
 *
 * 2. tiled: Shared memory + barriers + loops with high compute/memory ratio
 *    -> Pattern: GEMM, conv-like tiling strategies
 *    -> Strength: Data reuse via shared memory
 *
 * 3. reduction: Shared memory + barriers with modest compute/memory (between tiled & elementwise)
 *    -> Pattern: Parallel reductions (sum, max, etc.)
 *    -> Strength: Synchronization-based cooperation
 *
 * 4. elementwise: No shared memory/sync, low compute/memory ratio
 *    -> Pattern: Per-element operations (ReLU, scale, etc.)
 *    -> Strength: Simple, predictable memory access
 *
 * 5. compute_heavy: High compute/memory without shared memory/sync
 *    -> Pattern: Dense computation (FFT, BLAS level-2, etc.)
 *    -> Strength: Computation-bound (not memory-bound)
 *
 * 6. irregular: Mixed signals
 *    -> Pattern: Data-dependent, complex control flow
 *    -> Challenge: Difficult to optimize generically
 *
 * Diagnostic signals (computed alongside classification):
 * Group A/C — Inefficiency signals derived from SASS instruction widths and counts:
 *   spill_severity            : fraction of instructions that are LDL/STL spill traffic
 *   store_uncoalesced_risk    : >75% of typed stores are 32-bit (write-side coalescing gap)
 *   store_vectorization_score : weighted STG width efficiency (0.25 = all 32-bit, 1.0 = all 128-bit)
 *   tensor_utilization_fraction: tensor_ops / (arithmetic + tensor + 1)
 *   productive_instruction_fraction: (compute + global I/O) / total_instructions
 *   shared_reuse_per_barrier  : shared_loads / (barriers + 1) — reuse between sync phases
 *   warp_divergence_risk      : (branches × loops) / (compute + 1) > 0.01
 *   fp_to_int_ratio           : (arithmetic_ops - integer_ops) / (integer_ops + 1)
 *
 * Groups E-H extend the result with stall-reason inference, warp-primitive
 * detection, memory-subsystem hazards, and a high-level kernel archetype.
 *
 * Micro-patterns provide detailed characteristics for deeper analysis.
 */

import type { PtxInstructionFeatures } from "./ptx_features";
import type { SassInstructionFeatures } from "./sass_features";
import { safeDiv } from "./opcode_kinds";

/**
 * Complete pattern analysis result for a kernel.
 * Combines pattern classification with detailed metrics and confidence.
 */
export interface PatternResult {
  class: string;                              // e.g., "tiled", "reduction", "elementwise", etc.
  confidence: number;                         // 0.0-1.0 reliability of classification
  shared_ops: number;                         // Shared memory operations
  global_ops: number;                         // Global memory operations
  barriers: number;                           // __syncthreads() calls
  branches: number;                           // Branch instructions
  loops: number;                              // Loop structures
  shared_to_global_ratio: number;             // Data reuse via shared memory
  branch_density: number;                     // branches / compute_ops
  branch_per_global_mem_op: number;           // branches / memory_ops (control intensity)
  barrier_density: number;                    // barriers / compute_ops (sync intensity)
  work_per_barrier: number;                   // compute / barriers (sync efficiency)
  compute_to_memory_ratio: number;            // FLOPs / memory_ops (key determinant)
  uses_tensor_cores: boolean;                 // SASS tensor op count > 0
  high_looping: boolean;                      // Loop density > 0.05
  sync_heavy: boolean;                        // Barrier density > 0.02
  control_irregular: boolean;                 // Branch + memory intensity > 0.15
  control_dominated: boolean;                 // Compute-to-branches ratio < 5.0
  complex_kernel: boolean;                    // 2+ of: loops, barriers, high branching
  tensor_dominated: boolean;                  // Uses tensor cores with high compute/memory ratio
  streaming: boolean;                         // Streaming memory access pattern
  sync_efficiency: string;                    // Quality of synchronization usage
  interleaving: string;                       // Load instruction grouping pattern
  /** True when SASS contains LDL/STL ops — register file was exhausted and spilled. */
  spill_risk: boolean;
  /** 0-1 spill traffic share across all instructions: (LDL+STL)/total_instructions. */
  spill_severity: number;
  /** True when >75% of typed loads are 32-bit — suggests uncoalesced memory access. */
  uncoalesced_risk: boolean;
  /** True when >75% of typed stores are 32-bit — write-side coalescing risk. */
  store_uncoalesced_risk: boolean;
  /** True when kernel is compute-heavy with no tensor ops — HMMA opportunity exists. */
  missing_tensor_cores: boolean;
  /** ATOM/RED > 5% of global mem ops — L2 serialisation risk in histogram/scatter kernels. */
  atomic_contention_risk: boolean;
  /** MUFU ops > 15% of arithmetic — SFU pipeline may be the throughput bottleneck. */
  sfu_heavy: boolean;
  /** 0.25–1.0: fraction of max vectorized load width; 1.0 = all LDG.128, 0.25 = all LDG.32. */
  vectorization_score: number;
  /** 0.25–1.0 write-side typed-width score; 1.0 = all STG.128, 0.25 = all STG.32. */
  store_vectorization_score: number;
  /** >1.5 barriers per loop iteration — likely syncing redundantly inside the loop body. */
  over_synchronized: boolean;
  /** Heavy scalar FP16 (HFMA/HADD/HMUL) with no HMMA — tensor-core FP16 opportunity. */
  fp16_scalar_risk: boolean;
  /** global_stores ≈ global_loads with low compute — scatter/histogram RMW shape. */
  read_modify_write: boolean;
  /** Tensor op fraction of compute instructions (0..1). */
  tensor_utilization_fraction: number;
  /** Fraction of total instructions that are "useful" compute or global memory work. */
  productive_instruction_fraction: number;
  /** Shared-load reuse between synchronization phases. */
  shared_reuse_per_barrier: number;
  /** Loop-aware divergence risk: (branches * loops)/(compute + 1) > 0.01. */
  warp_divergence_risk: boolean;
  /** FP-vs-int mix: (arithmetic_ops - integer_ops)/(integer_ops + 1). */
  fp_to_int_ratio: number;

  // ── Group E — Stall Reason Inference ──────────────────────────────────────
  /** Consecutive-load chain stall — latency-bound memory access (E1). */
  stall_memory_dependency: boolean;
  /** High global traffic + good interleaving + low compute — bandwidth-bound stall (E2). */
  stall_memory_throttle: boolean;
  /** Register spill rate > 3% of instructions — local-memory reload stall risk (E3). */
  stall_local_memory: boolean;
  /** barriers ≥ 2 + work_per_barrier < 30 + shared present — sync-dominated execution (E4). */
  stall_sync: boolean;

  // ── Group F — Warp-Level Primitives ────────────────────────────────────────
  /** SHFL instructions present — kernel uses warp shuffle communication (F2). */
  uses_warp_shuffle: boolean;
  /** VOTE / MATCH instructions present — kernel uses warp-vote predicates (F3). */
  uses_warp_vote: boolean;
  /** SHFL + barriers + no SFU — inferred warp-shuffle cooperative reduction (F4). */
  warp_reduction_pattern: boolean;

  // ── Group H — Kernel Archetype ─────────────────────────────────────────────
  /**
   * High-level kernel shape inferred from combined signals.
   * Values: `"gemm"` | `"attention"` | `"softmax"` | `"reduction"` |
   *         `"histogram_scatter"` | `"activation"` | `undefined`.
   * `undefined` when no archetype pattern is confidently matched.
   */
  archetype: string | undefined;

  source: string;                             // "sass" if SASS used, else "ptx"
  insight: string;                            // Human-readable explanation
  /**
   * B9: true when PTX instruction features were summed across 2+ `.entry` kernels.
   * When true, density-based signals (warp_divergence_risk, over_synchronized,
   * high_looping) are forced to false because loop/branch counts span multiple
   * kernels and cannot represent any individual kernel's ratios reliably.
   */
  multi_kernel_ptx: boolean;
}

/**
 * Classifies a kernel's computational pattern based on instruction features.
 *
 * Algorithm:
 * 1. Extract memory, compute, control, and synchronization metrics from PTX/SASS
 * 2. Calculate derived metrics: intensity ratios, density measures, efficiency scores
 * 3. Build boolean flags for Group A/C micro-pattern signals (spill, coalescing,
 *    vectorization, tensor utilization, divergence, productive-instruction fraction)
 * 4. Classify into primary pattern based on decision tree:
 *    - High branching -> control_heavy
 *    - Shared + barriers + loops + high compute -> tiled (or reduction if moderate compute)
 *    - No shared/barriers + low compute -> elementwise
 *    - No shared/barriers + high compute -> compute_heavy
 *    - Otherwise -> irregular
 * 5. Synthesize Groups E-H stall/primitive/hazard/archetype signals
 * 6. Boost confidence if SASS data available
 *
 * Key metrics:
 * - compute_to_memory = FLOPs / global_mem_ops (threshold ~2-4 distinguishes patterns)
 * - branch_density = branches / compute_ops (threshold ~0.1 for control-heavy)
 * - barrier_density = barriers / compute_ops (threshold ~0.02 for sync-heavy)
 * - work_per_barrier = compute / barriers (efficiency metric: how much work between syncs)
 * - spill_severity = (local_loads + local_stores) / total_instructions
 * - store_vectorization_score = weighted STG width / max-width baseline
 * - tensor_utilization_fraction = tensor_ops / (arithmetic + tensor + 1)
 * - productive_instruction_fraction = (compute + global I/O) / total_instructions
 * - shared_reuse_per_barrier = shared_loads / (barriers + 1)
 * - warp_divergence_risk = (branches * loops) / (compute + 1) > 0.01
 * - fp_to_int_ratio = (arithmetic_ops - integer_ops) / (integer_ops + 1)
 *
 * @param ptxFeatures High-level PTX instruction counts
 * @param sassFeatures Low-level SASS instruction counts (more accurate, optional)
 * @returns Pattern classification with metrics and confidence
 */
export function analyzePattern(
  ptxFeatures: PtxInstructionFeatures,
  sassFeatures?: SassInstructionFeatures
): PatternResult {
  // Extract SASS metrics if available (preferred over PTX estimates)
  let sassGlobal = 0;
  let sassShared = 0;
  let sassBarriers = 0;
  let sassBranches = 0;
  let sassCompute = 0;

  if (sassFeatures !== undefined) {
    sassGlobal = sassFeatures.global_loads + sassFeatures.global_stores;
    sassShared = sassFeatures.shared_loads + sassFeatures.shared_stores;
    sassBarriers = sassFeatures.barrier;
    sassBranches = sassFeatures.branch;
    sassCompute = sassFeatures.arithmetic_ops + sassFeatures.tensor_ops;
  }

  // PTX metrics (used if SASS not available)
  const ptxShared = 0;  // PTX lacks shared memory instructions
  const ptxBarriers = ptxFeatures.barrier;
  const ptxBranches = ptxFeatures.branches;
  const ptxLoops = ptxFeatures.loops;

  // B5 fix: switch on data-source presence, NOT on count > 0.
  // The old "sassCount > 0 ? sass : ptx" rule treated a legitimate zero SASS
  // count (e.g. no global loads in a pure shared-memory kernel) as "SASS has no
  // data" and silently substituted a stale PTX value.  That corrupted globalOps,
  // computeOps, barriers, and branches for any kernel where SASS was available
  // but one of those groups genuinely is zero.
  //
  // The correct rule: if SASS was provided, always use SASS (even when zero).
  // Only fall back to PTX when SASS was not provided at all.
  const hasSass = sassFeatures !== undefined;

  const sharedOps = hasSass ? sassShared : ptxShared;
  const barriers  = hasSass ? sassBarriers : ptxBarriers;
  const branches  = hasSass ? sassBranches : ptxBranches;
  const loops = ptxLoops;  // always from PTX — SASS has no loop-edge semantics

  const globalOps = hasSass
    ? sassGlobal
    : ptxFeatures.global_loads + ptxFeatures.global_stores;

  const computeOps = hasSass
    ? sassCompute
    : ptxFeatures.fma + ptxFeatures.add + ptxFeatures.mul;

  const usesTensor =
    sassFeatures !== undefined && sassFeatures.tensor_ops > 0;

  // Key ratio metrics for pattern classification
  const sharedToGlobal = safeDiv(sharedOps, globalOps);  // Reuse effectiveness
  const branchDensity = safeDiv(branches, computeOps + 1);  // Branch frequency vs compute
  const branchPerMem = safeDiv(branches, globalOps + 1);  // Branch intensity vs memory
  const computeToMemory = safeDiv(computeOps, globalOps + 1);  // Primary classifier

  // Boolean flags for micro-patterns
  const hasShared = sharedOps > 0;
  const hasBarrier = barriers > 0;
  const hasLoops = loops > 0;
  // Control-heavy heuristic: high branch density OR high branch vs memory with low compute
  const highBranching =
    branchDensity > 0.1 ||
    (branchPerMem > 0.08 && computeToMemory < 4.0);

  const loopDensity = safeDiv(loops, computeOps + 1);
  const barrierDensity = safeDiv(barriers, computeOps + 1);
  const controlIntensity = branchDensity + branchPerMem;
  const computeVsControl = safeDiv(computeOps, branches + 1);
  
  // Complexity = how many structural features are present
  const structuralComplexity =
    (hasLoops ? 1 : 0) + (hasBarrier ? 1 : 0) + (highBranching ? 1 : 0);
  
  // Tensor pattern: uses tensor ops AND has high compute/memory ratio
  const tensorDominated = usesTensor && computeToMemory > 4.0;

  // Micro-pattern flags
  const highLooping = loopDensity > 0.05;
  const syncHeavy = barrierDensity > 0.02;
  const controlIrregular = controlIntensity > 0.15;
  const controlDominated = computeVsControl < 5.0;
  const complexKernel = structuralComplexity >= 2;
  
  // Streaming pattern: no shared memory/sync/loops but low compute ratio (I/O bound)
  const isStreaming =
    !hasShared && !hasBarrier && !hasLoops && computeToMemory < 2.0;
  
  // Synchronization efficiency: amount of work between barriers
  const workPerBarrier = safeDiv(computeOps, barriers + 1);
  let syncEfficiency: string;
  if (workPerBarrier > 100) {
    syncEfficiency = "efficient";      // Lots of work between syncs
  } else if (workPerBarrier > 30) {
    syncEfficiency = "moderate";       // Reasonable work distribution
  } else {
    syncEfficiency = "inefficient";    // Too many barriers
  }

  // Memory access interleaving patterns (from SASS if available)
  const interleaveScore =
    sassFeatures !== undefined ? sassFeatures.stream_interleave_score : 0;
  const maxLoadRun =
    sassFeatures !== undefined
      ? sassFeatures.stream_max_consecutive_loads
      : 0;
  let interleavingPattern: string;
  if (interleaveScore > 0.4) {
    interleavingPattern = "interleaved";  // Mixed load/store instructions
  } else if (maxLoadRun > 4) {
    interleavingPattern = "stacked";      // Many loads in a row
  } else {
    interleavingPattern = "mixed";        // Default
  }

  // ── Inefficiency signals ────────────────────────────────────────────────────────
  //
  // Spill risk: any LDL (load local) or STL (store local) traffic proves the
  // compiler had to evict live values from the 64 K-register file to per-thread
  // local memory (backed by L1/L2/DRAM). Even a handful of LDL/STL per thread
  // stalls a warp for 100+ cycles and forces extra global-memory bandwidth.
  const spillRisk =
    sassFeatures !== undefined &&
    (sassFeatures.local_loads + sassFeatures.local_stores) > 0;

  const totalInstrCount =
    sassFeatures !== undefined ? sassFeatures.total_instructions : 0;
  const localOpsCount =
    sassFeatures !== undefined
      ? sassFeatures.local_loads + sassFeatures.local_stores
      : 0;
  const spillSeverity =
    totalInstrCount > 0 ? safeDiv(localOpsCount, totalInstrCount) : 0;

  // Uncoalesced access risk (SASS required):
  // When ≥75% of typed (known-width) global loads are narrow 32-bit ops AND
  // there are enough typed loads to be statistically meaningful (> 4), the
  // threads in a warp likely address non-contiguous 32-bit words. The memory
  // controller cannot merge those into a single 128-byte transaction, so it
  // issues one sub-transaction per thread — a warp-serialisation bottleneck.
  const totalLdgTyped =
    sassFeatures !== undefined
      ? sassFeatures.ldg_128 + sassFeatures.ldg_64 + sassFeatures.ldg_32
      : 0;
  const narrowLoadRatio =
    totalLdgTyped > 4
      ? safeDiv(sassFeatures!.ldg_32, totalLdgTyped)
      : 0;
  const uncoalescedRisk = narrowLoadRatio > 0.75 && sassGlobal > 4;

  const totalStgTyped =
    sassFeatures !== undefined
      ? sassFeatures.stg_128 + sassFeatures.stg_64 + sassFeatures.stg_32
      : 0;
  const narrowStoreRatio =
    totalStgTyped > 4
      ? safeDiv(sassFeatures!.stg_32, totalStgTyped)
      : 0;
  const storeUncoalescedRisk =
    narrowStoreRatio > 0.75 && totalStgTyped > 4 && globalOps > 4;

  // Missing tensor-core opportunity:
  // A kernel is flagged when it is compute-heavy (compute_to_memory > 4) and
  // does substantial arithmetic (computeOps > 16) entirely with scalar FFMA
  // lanes, with no MMA/HMMA/WGMMA instructions at all. On SM ≥ 7.0 (Volta+)
  // these workloads (GEMM, dot products, convolutions) can use HMMA for
  // 4–16× higher throughput.
  const missingTensorCores =
    !usesTensor &&
    computeToMemory > 4.0 &&
    computeOps > 16;

  // ── Atomic contention risk ────────────────────────────────────────────────
  //
  // ATOM / RED instructions on global memory serialize through the L2 cache
  // partition that owns the target address.  The GPU issues each atomic as an
  // exclusive read-modify-write against a 128-byte cache line: all concurrent
  // requests queue behind a mutex in the memory controller, turning O(warps)
  // parallel threads into O(warps) sequential transactions.
  //
  // Threshold: 5% of global memory ops being atomics is the practical point at
  // which the serialization latency begins to dominate kernel runtime.  Beyond
  // ~10% the kernel is almost certainly atomic-bottlenecked.
  //
  // Fix strategies: per-thread private histograms (privatization), warp-level
  // SIMT reduction before the atomic, or replacing ATOM.ADD with a warp shuffle
  // reduction + a single atomic per warp.
  // Use global_atomic_ops (ATOM + RED only, not ATOMS) so that kernels using
  // shared-memory atomics heavily are not falsely flagged for L2 contention.
  // ATOMS causes shared-bank serialisation, not L2 — a different bottleneck.
  const globalAtomicOps =
    sassFeatures !== undefined ? sassFeatures.global_atomic_ops : 0;
  const atomicContentionRisk =
    globalAtomicOps > 0 && globalOps > 0 && safeDiv(globalAtomicOps, globalOps) > 0.05;

  // ── SFU throughput bottleneck ─────────────────────────────────────────────
  //
  // The Special Function Unit executes MUFU.SIN, MUFU.COS, MUFU.EXP2,
  // MUFU.LOG2, MUFU.RCP, MUFU.RSQ, MUFU.SQRT (one instruction per sinf/cosf/
  // expf/logf/rcpf/rsqrtf/sqrtf call in C code).  The SFU pipeline has exactly
  // 1/4 the throughput of the main FP32 ALU per SM.
  //
  // Effect: if 15%+ of arithmetic instructions go through the SFU, the warp
  // scheduler must stall four cycles for every SFU op while the FP32 lanes
  // sit idle.  The bottleneck is invisible to memory-bandwidth analysis.
  //
  // Fix: replace MUFU calls with polynomial approximations (__expf → exp2f +
  // multiply) or restructure to amortize SFU cost across more FP32 work per
  // dispatch.
  const sfuOps =
    sassFeatures !== undefined ? sassFeatures.sfu_ops : 0;
  const sfuHeavy =
    sfuOps > 8 && safeDiv(sfuOps, computeOps + sfuOps + 1) > 0.15;

  // ── Vectorization score ───────────────────────────────────────────────────
  //
  // A fully vectorized kernel loads four contiguous FP32 values in a single
  // LDG.128 (128-bit) instruction.  Scalar (LDG.32) loads the same data in
  // four transactions, quadrupling L2 bandwidth pressure and instruction
  // overhead without changing the amount of computation.
  //
  // Score formula: compute a weighted average width in 32-bit lanes:
  //   weightedLanes = ldg_128×4 + ldg_64×2 + ldg_32×1
  //   totalLanes    = totalTyped × 4  (max possible is 4 lanes per instruction)
  //   score         = weightedLanes / totalLanes
  //
  // Interpretation:
  //   1.00 — 100% LDG.128: perfectly vectorized, each warp issues one transaction
  //   0.50 — mix of 128-bit and 32-bit loads
  //   0.25 — 100% LDG.32: fully scalar, 4× the memory transactions necessary
  //
  // Returns 0 when SASS is unavailable or when fewer than 4 typed loads exist
  // (too few data points for the ratio to be meaningful).
  let vectorizationScore = 0;
  if (totalLdgTyped >= 4 && sassFeatures !== undefined) {
    const weightedLanes =
      sassFeatures.ldg_128 * 4 +
      sassFeatures.ldg_64  * 2 +
      sassFeatures.ldg_32  * 1;
    vectorizationScore =
      Math.round(safeDiv(weightedLanes, totalLdgTyped * 4) * 1e6) / 1e6;
  }

  let storeVectorizationScore = 0;
  if (totalStgTyped >= 4 && sassFeatures !== undefined) {
    const weightedStoreLanes =
      sassFeatures.stg_128 * 4 +
      sassFeatures.stg_64 * 2 +
      sassFeatures.stg_32 * 1;
    storeVectorizationScore =
      Math.round(safeDiv(weightedStoreLanes, totalStgTyped * 4) * 1e6) / 1e6;
  }

  // ── Over-synchronization ─────────────────────────────────────────────────
  //
  // __syncthreads() (BAR.SYNC) is the CUDA mechanism for ensuring all threads
  // in a block have finished writing shared memory before any thread reads it.
  // The correct usage pattern is one barrier per cooperative phase:
  //   [load tile] → BAR.SYNC → [compute] → BAR.SYNC → [load next tile] → ...
  //
  // A common mistake is placing BAR.SYNC inside a loop body where only one sync
  // per outer iteration is needed. Each extra BAR.SYNC:
  //   1. Drains the warp scheduler: all 32 threads must reach the barrier before
  //      any thread can continue — stalling the entire block for L1/shared latency.
  //   2. Prevents instruction-level parallelism across the barrier point.
  //
  // Heuristic: if the barrier / loop ratio exceeds 1.5, the kernel almost
  // certainly syncs more than once per loop iteration. Require at least 2
  // barriers and at least 1 loop to avoid false positives on trivially simple
  // kernels.
  const overSynchronized =
    loops > 0 &&
    barriers >= 2 &&
    safeDiv(barriers, loops) > 1.5;

  // ── FP16 scalar risk ─────────────────────────────────────────────────────
  //
  // HFMA / HADD / HMUL are scalar FP16 operations that run on the CUDA core
  // FP16 pipeline at 2× the throughput of FP32 FFMA (two FP16 ops per clock
  // per CUDA core vs. one FP32 op).  However, on SM ≥ 7.0 (Volta and later)
  // the Tensor Core HMMA instruction performs a full 16×16×16 FP16 matrix
  // multiply using the dedicated matrix units, delivering approximately
  // 16–32× the throughput of scalar HFMA.
  //
  // A kernel that does heavy FP16 arithmetic but emits zero HMMA / WGMMA / WMMA
  // instructions is missing the tensor-core path entirely — this typically
  // happens when the programmer uses element-wise FP16 arrays instead of the
  // WMMA / cuBLAS / cuDNN APIs.
  //
  // Note: This is separate from `missing_tensor_cores` (which targets FP32
  // FFMA heavy kernels) so that both FP32 and FP16 missed-TC opportunities are
  // flagged independently.
  const fp16ArithOps =
    sassFeatures !== undefined ? sassFeatures.fp16_arith_ops : 0;
  const fp16ScalarRisk =
    fp16ArithOps > 8 &&
    (sassFeatures?.wmma_ops ?? 0) === 0 &&
    safeDiv(fp16ArithOps, computeOps + 1) > 0.2;

  // ── Group E — Stall Reason Inference ───────────────────────────────────────────

  // E1: consecutive-load chain — latency-bound stall pattern.
  // Long runs of back-to-back loads with no intervening compute mean the warp
  // stalls on each load's response before issuing the next request.  This is
  // the difference between latency-bound (E1) and bandwidth-bound (E2).
  const stallMemoryDependency =
    sassFeatures !== undefined &&
    sassFeatures.stream_max_consecutive_loads > 4 &&
    sassFeatures.stream_interleave_score < 0.2;

  // E2: bandwidth-saturating stall.
  // Good interleaving (loads and compute alternate) but more memory ops than
  // compute work means the L2/DRAM pipeline is perpetually full.
  // Uses computeToMemory < 2.0 as a proxy for low arithmetic intensity.
  const stallMemoryThrottle =
    sassFeatures !== undefined &&
    globalOps > 16 &&
    sassFeatures.stream_interleave_score > 0.4 &&
    computeToMemory < 2.0;

  // E3: local-memory spill stall.
  // Compute what fraction of all instructions are spill I/O (LDL + STL).
  // > 3% means the warp scheduler is regularly blocked waiting for
  // per-thread local-memory reloads at 100–600 cycle latency.
  // Note: occupancy cross-check (< 0.5) is applied in diagnoseKernel().
  const stallLocalMemory = spillSeverity > 0.03;

  // E4: sync-dominated execution.
  // When most instructions are separated by a barrier with little compute
  // between them, the block spends most time at __syncthreads() waits.
  const stallSync = barriers >= 2 && workPerBarrier < 30 && hasShared;

  // ── Group F — Warp-Level Primitives ───────────────────────────────────────────

  const warpShuffleOps =
    sassFeatures !== undefined ? sassFeatures.warp_shuffle_ops : 0;
  const warpVoteOps =
    sassFeatures !== undefined ? sassFeatures.warp_vote_ops : 0;
  // F2: kernel uses warp-shuffle communication.
  const usesWarpShuffle = warpShuffleOps > 0;
  // F3: kernel uses warp-vote predicates.
  const usesWarpVote = warpVoteOps > 0;
  // F4: inferred warp-shuffle cooperative reduction.
  // SHFL present + barriers present + no SFU ops (not a transcendental-heavy
  // kernel) = most likely doing a register-file reduction between tiles.
  const warpReductionPattern =
    warpShuffleOps > 0 && sfuOps === 0 && barriers > 0;

  const globalLoads =
    sassFeatures !== undefined ? sassFeatures.global_loads : ptxFeatures.global_loads;
  const globalStores =
    sassFeatures !== undefined ? sassFeatures.global_stores : ptxFeatures.global_stores;

  const tensorUtilizationFraction =
    Math.round(safeDiv(sassFeatures?.tensor_ops ?? 0, (sassFeatures?.arithmetic_ops ?? 0) + (sassFeatures?.tensor_ops ?? 0) + 1) * 1e6) / 1e6;
  const productiveInstructionFraction =
    totalInstrCount > 0
      ? Math.round(
          safeDiv(
            (sassFeatures?.arithmetic_ops ?? 0) +
              (sassFeatures?.tensor_ops ?? 0) +
              globalLoads +
              globalStores,
            totalInstrCount
          ) * 1e6
        ) / 1e6
      : 0;
  const sharedLoadsCount =
    sassFeatures !== undefined ? sassFeatures.shared_loads : 0;
  const sharedReusePerBarrier =
    Math.round(safeDiv(sharedLoadsCount, barriers + 1) * 1e6) / 1e6;
  const warpDivergenceRisk =
    safeDiv(branches * loops, computeOps + 1) > 0.01;
  const fpToIntRatio =
    sassFeatures !== undefined
      ? Math.round(
          safeDiv(
            sassFeatures.arithmetic_ops - sassFeatures.integer_ops,
            sassFeatures.integer_ops + 1
          ) * 1e6
        ) / 1e6
      : 0;

  //
  // Examples: histogram accumulation, sparse vector scatter, in-place prefix
  // sums, and any kernel that reads a value, applies a simple function (or
  // a lookup), and writes the result back to the same or nearby address.
  //
  // Why this matters:
  //   - Loads and stores to the same address cannot be pipelined by the memory
  //     controller; the store must wait for the load to complete.
  //   - If the write address depends on the loaded value (indirect scatter),
  //     the GPU cannot speculate the store address, blocking instruction-level
  //     parallelism.
  //   - Without atomics the pattern causes data races; with atomics it triggers
  //     atomic_contention_risk as well.
  //
  // Fix: privatization — give each warp a private copy of the accumulator in
  // shared memory, reduce within the warp/block, then issue a single atomic
  // write to global memory per warp.
  //
  // Thresholds:
  //   store_to_load_ratio in (0.5, 2.0) — stores and loads are comparable
  //   computeToMemory < 1.0             — very little arithmetic per byte moved
  //   globalOps > 4                     — avoid flagging trivial kernels
  const storeToLoadRatio = safeDiv(globalStores, globalLoads + 1);
  const readModifyWrite =
    storeToLoadRatio > 0.5 &&
    storeToLoadRatio < 2.0 &&
    computeToMemory < 1.0 &&
    globalOps > 4;
  // Pattern classification decision tree
  let pattern: string;
  let confidence: number;
  let insight: string;

  if (highBranching) {
    // Branch divergence is dominant constraint
    pattern = "control_heavy";
    confidence = 0.85;
    insight =
      "high branching (vs compute or memory) suggests control-flow dominated kernel";
  } else if (hasShared && hasBarrier && hasLoops) {
    // Shared memory + sync + loops suggests cooperative tiling pattern
    if (computeToMemory > 8) {
      // Very high compute/memory = GEMM/matrix-like pattern
      pattern = "tiled";
      confidence = 0.9;
      if (usesTensor) {
        insight =
          "high compute vs global memory with tiling and tensor ops suggests " +
          "GEMM/conv-like pattern";
      } else {
        insight =
          "shared memory, sync, and loops with high compute/memory ratio suggest tiling";
      }
    } else {
      // Moderate compute/memory = reduction-like pattern
      pattern = "reduction";
      confidence = 0.82;
      insight =
        "shared memory, barriers, and loops with modest compute/memory ratio " +
        "suggest reduction";
    }
  } else if (!hasShared && !hasBarrier) {
    // No cooperation -> either elementwise or compute-heavy
    if (computeToMemory < 2) {
      // Low compute/memory = per-element operations
      pattern = "elementwise";
      confidence = 0.8;
      insight =
        "low compute vs global traffic without shared/sync suggests elementwise work";
    } else {
      // High compute/memory = dense computation
      pattern = "compute_heavy";
      confidence = 0.78;
      insight =
        "high compute vs global ops without shared/sync suggests compute-bound kernel";
    }
  } else {
    // Mixed signals = hard to classify
    pattern = "irregular";
    confidence = 0.6;
    insight = "mixed signals indicate irregular or data-dependent behavior";
  }

  // Boost confidence if SASS data available (more accurate)
  if (sassFeatures !== undefined) {
    confidence += 0.1;
  }
  confidence = Math.min(confidence, 0.95);  // Cap at 0.95 (never 100% certain)

  // B9: When PTX features are a sum across multiple .entry kernels, loop and
  // branch counts cross kernel boundaries and cannot represent any individual
  // kernel's density ratios reliably.  Force the affected booleans to false so
  // they do not mislead the diagnosis or UI badge layer.
  const isMultiKernelPtx = !hasSass && ptxFeatures.kernelCount > 1;
  const safeHighLooping      = isMultiKernelPtx ? false : highLooping;
  const safeOverSynchronized = isMultiKernelPtx ? false : overSynchronized;
  const safeWarpDivergence   = isMultiKernelPtx ? false : warpDivergenceRisk;
  const safeComplexKernel    = isMultiKernelPtx ? false : complexKernel;

  // ── Group H — Kernel Archetype Recognizers ─────────────────────────────────
  // Evaluated in order; first match wins.  Raw variables already in scope:
  //   pattern, usesTensor, sharedToGlobal (== shared/global ratio), computeToMemory,
  //   sfuOps, barriers, workPerBarrier, readModifyWrite, atomicContentionRisk.
  let archetype: string | undefined;
  // H4: histogram / scatter (most specific: RMW + global atomics)
  if (readModifyWrite && atomicContentionRisk && computeToMemory < 1.0) {
    archetype = "histogram_scatter";
  // H1: GEMM (tiled + tensor + high reuse + high compute)
  } else if (
    pattern === "tiled" &&
    usesTensor &&
    sharedToGlobal > 2.0 &&
    computeToMemory > 8.0 &&
    sharedReusePerBarrier >= 8
  ) {
    archetype = "gemm";
  // H2: attention (reduction + SFU + tensor) / softmax (reduction + SFU only)
  } else if (
    pattern === "reduction" &&
    sfuOps > 4 &&
    barriers >= 2 &&
    workPerBarrier < 50
  ) {
    archetype = usesTensor ? "attention" : "softmax";
  // H3: parallel reduction (reduction + shared + low compute, no tensor)
  } else if (
    pattern === "reduction" &&
    barriers >= 2 &&
    sharedOps > 0 &&
    computeToMemory < 4.0 &&
    !usesTensor
  ) {
    archetype = "reduction";
  // H5: elementwise activation (elementwise + SFU, no tensor)
  } else if (pattern === "elementwise" && sfuOps > 0 && !usesTensor) {
    archetype = "activation";
  }

  return {
    class: pattern,
    confidence: Math.round(confidence * 100) / 100,
    shared_ops: sharedOps,
    global_ops: globalOps,
    barriers,
    branches,
    loops,
    shared_to_global_ratio: Math.round(sharedToGlobal * 1e6) / 1e6,
    branch_density: Math.round(branchDensity * 1e6) / 1e6,
    branch_per_global_mem_op: Math.round(branchPerMem * 1e6) / 1e6,
    barrier_density: Math.round(barrierDensity * 1e6) / 1e6,
    work_per_barrier: Math.round(workPerBarrier * 1e6) / 1e6,
    compute_to_memory_ratio: Math.round(computeToMemory * 1e6) / 1e6,
    uses_tensor_cores: usesTensor,
    streaming: isStreaming,
    sync_efficiency: syncEfficiency,
    interleaving: interleavingPattern,
    spill_risk: spillRisk,
    spill_severity: Math.round(spillSeverity * 1e6) / 1e6,
    uncoalesced_risk: uncoalescedRisk,
    store_uncoalesced_risk: storeUncoalescedRisk,
    missing_tensor_cores: missingTensorCores,
    atomic_contention_risk: atomicContentionRisk,
    sfu_heavy: sfuHeavy,
    vectorization_score: vectorizationScore,
    store_vectorization_score: storeVectorizationScore,
    fp16_scalar_risk: fp16ScalarRisk,
    read_modify_write: readModifyWrite,
    tensor_utilization_fraction: tensorUtilizationFraction,
    productive_instruction_fraction: productiveInstructionFraction,
    shared_reuse_per_barrier: sharedReusePerBarrier,
    fp_to_int_ratio: fpToIntRatio,
    stall_memory_dependency: stallMemoryDependency,
    stall_memory_throttle: stallMemoryThrottle,
    stall_local_memory: stallLocalMemory,
    stall_sync: stallSync,
    uses_warp_shuffle: usesWarpShuffle,
    uses_warp_vote: usesWarpVote,
    warp_reduction_pattern: warpReductionPattern,
    archetype,
    high_looping: safeHighLooping,
    sync_heavy: syncHeavy,
    control_irregular: controlIrregular,
    control_dominated: controlDominated,
    complex_kernel: safeComplexKernel,
    tensor_dominated: tensorDominated,
    over_synchronized: safeOverSynchronized,
    warp_divergence_risk: safeWarpDivergence,
    source: sassFeatures !== undefined ? "sass" : "ptx",
    insight,
    multi_kernel_ptx: isMultiKernelPtx,
  };
}
