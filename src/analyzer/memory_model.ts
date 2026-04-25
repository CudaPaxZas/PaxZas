/**
 * Memory intensity model (matches memory_model.py).
 *
 * This module analyzes GPU kernel memory behavior to determine:
 * - How "memory-bound" a kernel is (does memory movement limit performance?)
 * - Arithmetic intensity (computations per byte transferred)
 * - Data reuse patterns (shared memory vs global memory traffic)
 * - Memory access efficiency (streaming, cache behavior)
 *
 * The analysis combines PTX (high-level IR) and SASS (low-level assembly) instruction counts
 * to classify kernels and identify optimization opportunities. Key classifications:
 * - compute_friendly: High compute/memory ratio; not memory-limited
 * - memory_bound: Low compute/memory ratio; memory bandwidth is the bottleneck
 * - reuse_optimized: Effective use of shared memory for data reuse
 * - balanced: Mixed characteristics without strong imbalance
 *
 * Additional Group B signals derived from SASS store widths:
 * - store_vectorization_score : weighted STG width efficiency (0.25 = all 32-bit, 1.0 = all 128-bit)
 * - load_store_ratio           : global_loads / (global_stores + 1)
 * - load_store_balance         : "read_dominated" (>4.0) | "write_dominated" (<0.5) | "balanced"
 */

import type { PtxInstructionFeatures } from "./ptx_features";
import { bytesHeuristic, flopsHeuristic } from "./ptx_features";
import type { SassInstructionFeatures } from "./sass_features";

/**
 * Safely divides two numbers, returning 0 if denominator is zero.
 * Prevents Infinity and NaN results from division by zero.
 */
function safeDiv(n: number, d: number): number {
  if (d === 0) {
    return 0;
  }
  return n / d;
}

/**
 * Estimates total bytes transferred by analyzing SASS load/store instructions.
 *
 * Strategy:
 * - ldg_128, ldg_64, ldg_32 are loads with known byte widths (128-bit, 64-bit, 32-bit)
 * - UnknownLdgs (global_loads - known_ldg_variants) are assumed to be defaultBytesPerOp
 * - Global stores use the default assumption
 *
 * This provides a reasonably accurate byte count when SASS is available.
 * Used to compute arithmetic intensity = FLOPs / bytes_moved.
 *
 * @param sass SASS instruction features with load/store counts
 * @param defaultBytesPerOp Default byte size for unknown load variants and all stores (typically 4)
 * @returns Estimated total bytes transferred
 */
function estimateSassBytes(
  sass: SassInstructionFeatures,
  defaultBytesPerOp: number
): number {
  // Loads: count typed widths; remaining unknowns fall back to defaultBytesPerOp
  const knownLdg = sass.ldg_128 + sass.ldg_64 + sass.ldg_32;
  const unknownLdg = Math.max(0, sass.global_loads - knownLdg);

  // Stores: Gap 4 fix — use typed widths instead of assuming 4 bytes for all stores.
  // Before this fix, a kernel emitting STG.E.128 (16-byte coalesced writes) would
  // have its write bandwidth undercounted by 4×, making arithmetic intensity appear
  // 2–4× higher than reality and causing near-miss mis-classifications.
  const knownStg = sass.stg_128 + sass.stg_64 + sass.stg_32;
  const unknownStg = Math.max(0, sass.global_stores - knownStg);

  return (
    sass.ldg_128 * 16 +              // 128-bit loads  = 16 bytes
    sass.ldg_64  * 8 +               //  64-bit loads  =  8 bytes
    sass.ldg_32  * 4 +               //  32-bit loads  =  4 bytes
    unknownLdg   * defaultBytesPerOp + // unknown loads = default
    sass.stg_128 * 16 +              // 128-bit stores = 16 bytes
    sass.stg_64  * 8 +               //  64-bit stores =  8 bytes
    sass.stg_32  * 4 +               //  32-bit stores =  4 bytes
    unknownStg   * defaultBytesPerOp   // unknown stores = default
  );
}

/**
 * Complete memory analysis result for a kernel.
 *
 * Fields document memory access patterns and compute intensity metrics.
 * The classification (class) determines memory optimization strategy:
 * - compute_friendly: Prioritize throughput (not memory-bound)
 * - memory_bound: Prioritize bandwidth (optimize memory access patterns)
 * - reuse_optimized: Leverage shared memory and cache hierarchy
 * - balanced: General optimizations apply
 */
export interface MemoryAnalysis {
  // Primary classification
  class: string;                 // "compute_friendly" | "memory_bound" | "reuse_optimized" | "balanced"

  // Instruction counts (merged from SASS if available, else PTX estimates)
  global_loads: number;          // Global (device) memory loads
  global_stores: number;         // Global memory stores
  shared_loads: number;          // Shared (local) memory loads
  shared_stores: number;         // Shared memory stores
  global_mem_ops: number;        // Total global memory operations
  shared_mem_ops: number;        // Total shared memory operations
  
  // Memory pressure metrics
  memory_pressure: number;       // Total bytes moved (high = memory bottleneck risk)
  
  // Data source attribution
  global_mem_source: string;     // "sass" if SASS data used, else "ptx"
  shared_mem_source: string;     // "sass" if SASS data used, else "unknown"
  compute_source: string;        // "sass" if SASS FLOPs used, else "ptx"
  
  // Cache/memory behavior
  cache_policy: string | null;   // "L2", "streaming", or null (not determinable)
  
  // Confidence score (0.0-1.0) indicating reliability of classification
  confidence: number;
  
  // Arithmetic intensity = FLOPs / bytes (key metric for performance prediction)
  // High intensity = compute-bound, low = memory-bound
  // null indicates infinite (zero bytes = pure compute)
  arithmetic_intensity_ops_per_byte: number | null;
  
  // Data reuse patterns
  reuse_ratio: number;           // shared_loads / global_loads (cache effectiveness)
  reuse_strength: number;        // shared_loads / global_mem_ops (reuse impact)
  mem_compute_ratio: number;     // bytes_moved / FLOPs (inverse of intensity)
  /** 0.25–1.0 write-side typed-width efficiency; 1.0 = all STG.128, 0.25 = all STG.32.
   *  Returns 0 when SASS is unavailable or fewer than 4 typed stores exist. */
  store_vectorization_score: number;
  /** global_loads / (global_stores + 1). High ratio = read-heavy; low = write-heavy. */
  load_store_ratio: number;
  /** "read_dominated" (ratio > 4.0) | "write_dominated" (ratio < 0.5) | "balanced". */
  load_store_balance: string;
  
  // Proxy values for compute and memory (used in classification heuristics)
  bytes_proxy: number;           // SASS-derived bytes or PTX estimate
  flops_proxy: number;           // max(SASS FLOPs, PTX FLOPs estimate)
  ptx_flops_proxy: number;       // PTX-only FLOP estimate
  sass_flops_proxy: number | null;  // SASS-only FLOP estimate (if available)
  ptx_bytes_proxy: number;       // PTX-only byte estimate
  
  // Human-readable explanation of classification
  insight: string;
}

/**
 * Analyzes memory intensity and data flow characteristics of a CUDA kernel.
 *
 * Algorithm:
 * 1. Extract memory instruction counts from PTX and SASS (SASS preferred when available)
 * 2. Compute arithmetic intensity = FLOPs / bytes (key performance metric)
 *    - SASS FLOPs account for tensor-core density (512 FLOPs/HMMA) and SFU ops
 *    - SASS byte estimate uses typed LDG/STG widths (16/8/4 bytes) for accuracy
 * 3. Measure data reuse via shared memory traffic
 * 4. Classify kernel based on intensity and reuse patterns
 * 5. Compute Group B store-side signals:
 *    - store_vectorization_score : weighted STG width / max-width baseline (0.25–1.0)
 *    - load_store_ratio           : global_loads / (global_stores + 1)
 *    - load_store_balance         : "read_dominated" | "write_dominated" | "balanced"
 * 6. Calculate confidence score combining data source quality
 *
 * Classification Logic:
 * - compute_friendly: Infinite intensity (no memory ops) or high intensity (>2)
 * - memory_bound: Low intensity (<0.5) with high bandwidth pressure
 * - reuse_optimized: Strong shared memory reuse (ratio > 2, strength > 1)
 * - balanced: Everything else
 *
 * The confidence score reflects:
 * - 0.6 base (heuristic estimates always have uncertainty)
 * + 0.25 if SASS available (more accurate than PTX estimates)
 * + 0.1 if high reuse or low intensity (clearer signal)
 * + 0.1 if streaming cache policy detected (LD.CS)
 * - 0.1/0.2 penalty if fewer than 5 global ops (insufficient data)
 * Then capped at 0.99 maximum.
 *
 * @param ptxFeatures High-level PTX instruction counts
 * @param sassFeatures Low-level SASS instruction counts (optional, more accurate)
 * @param bytesPerMemOp Default byte size for unknown memory ops (default 4 bytes)
 * @returns Complete MemoryAnalysis with classification and metrics
 */
export function analyzeMemory(
  ptxFeatures: PtxInstructionFeatures,
  sassFeatures: SassInstructionFeatures | undefined,
  bytesPerMemOp = 4
): MemoryAnalysis {
  // Extract PTX memory instruction counts
  const ptxGlobalLoads = ptxFeatures.global_loads;
  const ptxGlobalStores = ptxFeatures.global_stores;
  const ptxGlobalMemOps = ptxGlobalLoads + ptxGlobalStores;
  const ptxFlops = flopsHeuristic(ptxFeatures);  // Estimated FLOPs from PTX instructions

  // Extract SASS memory and compute instructions (if available; defaults to 0)
  let sassGlobalLoads = 0;
  let sassGlobalStores = 0;
  let sassSharedLoads = 0;
  let sassSharedStores = 0;
  let sassFlopsProxy = 0;
  let sassBytesProxy = 0;
  if (sassFeatures !== undefined) {
    sassGlobalLoads = sassFeatures.global_loads;
    sassGlobalStores = sassFeatures.global_stores;
    sassSharedLoads = sassFeatures.shared_loads;
    sassSharedStores = sassFeatures.shared_stores;
    // Gap 1: Fix tensor FLOP proxy.  The original formula used `2 × tensor_ops`
    // which gives ~2 FLOPs/instruction — correct for scalar FP32 FFMA but wildly
    // wrong for tensor ops.  A single HMMA.16816.F16 performs a 16×16×16 matrix
    // MAC across a warp = 8192 FLOPs; the per-instruction approximation is 512
    // (a widely-used practical figure).  Integer/binary MMA (IMMA/BMMA) uses 64
    // as a conservative floor.  Without this fix, a tensor-heavy kernel shows
    // near-zero arithmetic intensity → incorrectly classified as memory_bound.
    //
    // Gap 2: Include SFU ops.  MUFU.* instructions (sinf/cosf/expf/…) are real
    // arithmetic work (~4 float-op equivalents each) but were excluded from the
    // FLOP estimate, causing SFU-heavy kernels (sigmoid, GELU, rendering BVH) to
    // appear falsely memory-bound.
    const sassScalarFlops   = sassFeatures.arithmetic_ops * 2;
    const sassTensorFpFlops = sassFeatures.wmma_ops * 512;
    const sassTensorIntFlops =
      (sassFeatures.tensor_ops - sassFeatures.wmma_ops) * 64;
    const sassSfuFlops      = sassFeatures.sfu_ops * 4;
    sassFlopsProxy =
      sassScalarFlops + sassTensorFpFlops + sassTensorIntFlops + sassSfuFlops;
    // Detailed byte estimation from SASS load/store variant info
    sassBytesProxy = estimateSassBytes(sassFeatures, bytesPerMemOp);
  }

  // Prefer SASS data when available (more accurate), fall back to PTX estimates
  const hasSassGlobal = sassGlobalLoads + sassGlobalStores > 0;
  const hasSassShared = sassSharedLoads + sassSharedStores > 0;
  const globalLoads = hasSassGlobal ? sassGlobalLoads : ptxGlobalLoads;
  const globalStores = hasSassGlobal ? sassGlobalStores : ptxGlobalStores;
  const sharedLoads = hasSassShared ? sassSharedLoads : 0;    // PTX lacks shared memory counts
  const sharedStores = hasSassShared ? sassSharedStores : 0;

  // Aggregate memory operation counts
  const globalMemOps = globalLoads + globalStores;
  const sharedMemOps = sharedLoads + sharedStores;
  
  // Choose best FLOP estimate
  const flops = Math.max(sassFlopsProxy, ptxFlops);
  
  // Choose best byte estimate
  const bytesMoved = hasSassGlobal ? sassBytesProxy : globalMemOps * bytesPerMemOp;

  // Memory pressure = total bytes moved in kernel
  const memoryPressure = bytesMoved;

  // Arithmetic intensity = FLOPs per byte (key metric)
  // - High intensity (>2) = compute-bound (okay if memory-bounded)
  // - Low intensity (<0.5) = memory-bound (need bandwidth optimization)
  // - Infinity = pure compute (no memory ops)
  let arithmeticIntensity: number;
  if (bytesMoved <= 0) {
    arithmeticIntensity = flops > 0 ? Infinity : 0;
  } else {
    arithmeticIntensity = flops / bytesMoved;
  }

  // Data reuse metrics quantify shared memory effectiveness
  // - reuse_ratio = shared_loads / global_loads: how much reuse per load
  // - reuse_strength = shared_loads / global_mem_ops: impact of reuse on total traffic
  const reuseRatio = safeDiv(sharedLoads, globalLoads);
  const reuseStrength = safeDiv(sharedLoads, globalMemOps);
  
  // mem_compute_ratio = bytes / FLOPs (inverse of intensity)
  const memComputeRatio = safeDiv(bytesMoved, flops);
  const loadStoreRatio = safeDiv(globalLoads, globalStores + 1);

  let loadStoreBalance = "balanced";
  if (loadStoreRatio > 4.0) {
    loadStoreBalance = "read_dominated";
  } else if (loadStoreRatio < 0.5) {
    loadStoreBalance = "write_dominated";
  }

  let storeVectorizationScore = 0;
  if (sassFeatures !== undefined) {
    const totalStgTyped =
      sassFeatures.stg_128 + sassFeatures.stg_64 + sassFeatures.stg_32;
    if (totalStgTyped >= 4) {
      const weightedStoreLanes =
        sassFeatures.stg_128 * 4 +
        sassFeatures.stg_64 * 2 +
        sassFeatures.stg_32;
      storeVectorizationScore =
        Math.round(safeDiv(weightedStoreLanes, totalStgTyped * 4) * 1e6) / 1e6;
    }
  }

  // Detect cache policy from SASS cache control bits
  let cachePolicy: string | null = null;
  if (sassFeatures !== undefined) {
    // CG = cache global (loads go through L2 cache normally)
    // CS = cache streaming (cache bypasses, streaming loads)
    if (sassFeatures.cg_loads > 0) {
      cachePolicy = "L2";
    } else if (sassFeatures.cs_loads > 0) {
      cachePolicy = "streaming";
    }
  }
  const isStreaming = cachePolicy === "streaming";

  // Build confidence score (0.0-1.0) indicating classification reliability
  // Start with 0.6 base (heuristics have inherent uncertainty)
  let confidence = 0.6;
  if (sassFeatures !== undefined) {
    // SASS data is more accurate than PTX estimates, +0.25 confidence
    confidence += 0.25;
  }
  if (globalMemOps === 0) {
    // B6 fix: differentiate between "SASS confirmed zero" and "PTX reported zero".
    //
    // Old code: `confidence = 0.30` unconditionally.
    // Problem: this discarded the +0.25 SASS bonus accumulated above, so a SASS-
    // verified pure-compute kernel (arithmetic_ops > 0, zero global ops) received
    // the same low 0.30 score as a PTX-only guess.  SASS confirming zero global
    // ops is actually a strong, trustworthy signal — the hardware assembly has no
    // LDG/STG instructions.  The `compute_friendly` classification that follows is
    // more reliable when backed by SASS than any other data-source combination.
    //
    // New rule:
    //   - SASS present → zero is confirmed → keep accumulated score (≥ 0.85 with
    //     base + SASS bonus), cap at 0.99 as usual.
    //   - SASS absent  → PTX zero is unreliable (PTX may miss global ops that the
    //     compiler lowered to SASS LDG/STG) → override to 0.30.
    if (sassFeatures === undefined) {
      confidence = 0.3;
    }
    // When SASS is present, fall through — confidence already reflects SASS quality.
  } else {
    // Strong signals (clear reuse or clear memory-bound behavior) add 0.1
    if (
      reuseRatio > 2.0 ||
      (arithmeticIntensity !== Infinity && arithmeticIntensity < 0.5)
    ) {
      confidence += 0.1;
    }
    // Gap 5: Streaming access (LD.CS) is an unambiguous memory-bound signal
    // independent of intensity analysis — the programmer or compiler explicitly
    // chose to bypass the L1 cache.  Boost confidence rather than leaving it
    // at the base heuristic level.
    if (isStreaming) {
      confidence += 0.1;
    }
    // Gap 6: Very small kernels (< 5 global ops) have too few data points for
    // reliable classification.  A single stray load/store dominates the ratio.
    // Apply a confidence penalty proportional to data availability.
    if (globalMemOps < 5) {
      confidence -= sassFeatures !== undefined ? 0.1 : 0.2;
    }
  }
  confidence = Math.min(0.99, confidence);  // Cap at 0.99

  // Gap 3: Replace the hardcoded `highPressureThresholdBytes = 2048` with a
  // global-ops guard.  The fixed threshold excluded correctly-detected low-
  // intensity kernels that transferred < 2048 bytes (e.g. 96 bytes from
  // 24 LDG.32 instructions), falling them to the "balanced" bucket instead.
  // Requiring ≥ 4 global ops is a simple, size-independent safeguard that
  // avoids false-positives on 1–2 instruction micro-kernels while correctly
  // classifying any kernel with meaningful bandwidth pressure.
  let memClass: string;
  let insight: string;

  if (arithmeticIntensity === Infinity) {
    // Zero memory ops with nonzero compute = pure compute kernel
    memClass = "compute_friendly";
    insight = "no global bytes proxy with nonzero compute proxy";
  } else if (isStreaming && arithmeticIntensity < 0.5) {
    // Streaming access pattern (cache-bypassing) with low reuse = memory-bound
    memClass = "memory_bound";
    insight = "streaming access with cache-bypass (LD.CS) and low arithmetic intensity";
  } else if (arithmeticIntensity < 0.5 && globalMemOps >= 4) {
    // Low intensity with non-trivial bandwidth demand = primary bottleneck is memory
    memClass = "memory_bound";
    insight = "high bandwidth pressure";
  } else if (reuseRatio > 2.0 && reuseStrength > 1.0) {
    // Strong shared memory reuse pattern = optimized for local data reuse
    memClass = "reuse_optimized";
    insight = "shared-memory traffic indicates reuse-oriented behavior";
  } else if (arithmeticIntensity > 2.0) {
    // High compute/memory ratio = compute is more expensive than moving data
    memClass = "compute_friendly";
    insight = "compute-heavy intensity with moderate memory pressure";
  } else {
    // Mixed signals without strong imbalance
    memClass = "balanced";
    insight = "memory and compute proxies are not strongly imbalanced";
  }

  // Build and return complete analysis result with rounded metrics for readability
  return {
    class: memClass,
    global_loads: globalLoads,
    global_stores: globalStores,
    shared_loads: sharedLoads,
    shared_stores: sharedStores,
    global_mem_ops: globalMemOps,
    shared_mem_ops: sharedMemOps,
    memory_pressure: memoryPressure,
    global_mem_source: hasSassGlobal ? "sass" : "ptx",
    shared_mem_source: hasSassShared ? "sass" : "unknown",
    // B4: attribute to "sass" whenever SASS was present, even when arithmetic_ops = 0.
    // The old condition `sassFlopsProxy > 0` incorrectly reported "ptx" for pure-memory
    // or control-flow SASS kernels (no FP/INT instructions), contradicting the actual
    // data source and misleading the Raw Features tab provenance display.
    compute_source: sassFeatures !== undefined ? "sass" : "ptx",
    cache_policy: cachePolicy,
    confidence: Math.round(confidence * 100) / 100,
    arithmetic_intensity_ops_per_byte:
      arithmeticIntensity === Infinity
        ? null
        : Math.round(arithmeticIntensity * 1e6) / 1e6,
    reuse_ratio: Math.round(reuseRatio * 1e6) / 1e6,
    reuse_strength: Math.round(reuseStrength * 1e6) / 1e6,
    mem_compute_ratio: Math.round(memComputeRatio * 1e6) / 1e6,
    store_vectorization_score: storeVectorizationScore,
    load_store_ratio: Math.round(loadStoreRatio * 1e6) / 1e6,
    load_store_balance: loadStoreBalance,
    bytes_proxy: Math.round(bytesMoved * 1000) / 1000,
    flops_proxy: Math.round(flops * 1000) / 1000,
    ptx_flops_proxy: Math.round(ptxFlops * 1000) / 1000,
    sass_flops_proxy:
      sassFeatures !== undefined
        ? Math.round(sassFlopsProxy * 1000) / 1000
        : null,
    ptx_bytes_proxy: Math.round(
      bytesHeuristic(ptxFeatures, bytesPerMemOp) * 1000
    ) / 1000,
    insight,
  };
}
