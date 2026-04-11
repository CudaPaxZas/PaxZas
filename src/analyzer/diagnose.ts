/**
 * Cross-model bottleneck diagnosis (matches diagnose.py).
 *
 * This module is the fusion layer that combines outputs from the three
 * analytical models — MemoryAnalysis, KernelAnalysis (occupancy), and
 * PatternResult — to produce a single ranked bottleneck report.
 *
 * Design principles:
 * - Rules are priority-ordered: rule 1 is the most severe / certain.
 * - Every rule that fires contributes a tag + suggestion.
 * - The first match → primary_bottleneck; the rest → secondary_bottlenecks.
 * - No rule contradicts another: a kernel may have multiple concurrent issues.
 *
 * Signal sources used by the nine rules:
 *   Pattern model  : spill_risk, stall_memory_dependency, stall_memory_throttle,
 *                    atomic_contention_risk, read_modify_write, sfu_heavy,
 *                    stall_sync, uncoalesced_risk, store_uncoalesced_risk,
 *                    missing_tensor_cores, stall_local_memory
 *   Memory model   : class ("memory_bound" | "compute_friendly" | …)
 *   Occupancy model: occupancy, limiting_factor, register_pressure_margin,
 *                    next_occupancy_class
 *
 * New in Group B/D extensions:
 *   Rule 7 now fires on EITHER load-side (uncoalesced_risk) OR store-side
 *   (store_uncoalesced_risk) coalescing failures.
 *   primary_bottleneck defaults to "none_detected" when no rule fires.
 */

import type { MemoryAnalysis } from "./memory_model";
import type { KernelAnalysis } from "./occupancy_model";
import type { PatternResult } from "./pattern_model";

// ── G1: DiagnosisResult ────────────────────────────────────────────────────────

export interface DiagnosisResult {
  /** The single dominant bottleneck tag for this kernel. */
  primary_bottleneck: string;
  /** Additional contributing bottleneck tags, ordered by estimated impact. */
  secondary_bottlenecks: string[];
  /** Stall reason breakdown, fused from pattern + occupancy signals. */
  stall_profile: {
    memory_dependency: boolean;
    memory_throttle: boolean;
    /** Stall on per-thread local-memory spill I/O AND occupancy is low. */
    local_memory: boolean;
    sync: boolean;
  };
  /**
   * Ordered list of concrete optimisation suggestions corresponding to the
   * identified bottlenecks (primary first, then secondary).
   */
  optimization_priority: string[];
  /**
   * Fraction 0–1 expressing how strongly the model signals align.
   * 1.0 = multiple independent signals agree; 0.0 = no clear signal.
   */
  confidence: number;
}

// ── G2: diagnoseKernel ─────────────────────────────────────────────────────────

/**
 * Fuses three independent analysis models into a ranked bottleneck report.
 *
 * Algorithm:
 * 1. Evaluate the nine priority-ordered rules (each may add a tag+suggestion).
 * 2. The first match becomes `primary_bottleneck`; remaining matches become
 *    `secondary_bottlenecks` in the order they fired.
 * 3. Build `stall_profile` from direct pattern+occupancy signals.
 * 4. Set `confidence` = min(rules_fired / 3, 1.0).  Three or more independent
 *    signals agreeing gives full confidence; a single signal gives 0.33.
 * 5. Return the assembled `DiagnosisResult`.
 *
 * Rule summary:
 *   1. register_spill     — pattern.spill_risk AND occupancy < 0.4
 *   2. memory_latency     — memory_bound AND stall_memory_dependency (E1)
 *   3. memory_bandwidth   — memory_bound AND stall_memory_throttle (E2)
 *   4. atomic_contention  — atomic_contention_risk AND read_modify_write
 *   5. sfu_throughput     — sfu_heavy AND NOT memory_bound
 *   6. sync_overhead      — stall_sync (E4) AND tiled pattern
 *   7. uncoalesced_access — uncoalesced_risk OR store_uncoalesced_risk
 *   8. occupancy          — registers limiting AND occupancy < 0.4
 *   9. compute_bound      — compute_friendly AND missing_tensor_cores
 *
 * @param memory   Output of `analyzeMemory()` for this kernel.
 * @param occ      Output of `analyzeKernel()` for this kernel.
 * @param pattern  Output of `analyzePattern()` for this kernel.
 */
export function diagnoseKernel(
  memory: MemoryAnalysis,
  occ: KernelAnalysis,
  pattern: PatternResult
): DiagnosisResult {
  const occupancy = occ.occupancy ?? 0;

  // ── Inference rules (priority-ordered) ────────────────────────────────────
  //
  // Each rule is a cross-model predicate that combines signals from two or
  // more models.  All 9 rules are always evaluated; the first match sets the
  // primary_bottleneck; subsequent matches become secondary_bottlenecks.

  type Rule = { tag: string; suggestion: string };
  const matched: Rule[] = [];

  // Rule 1 — Register spill
  // Cross: pattern.spill_risk (LDL/STL present) × occupancy < 0.4.
  // Spilling to per-thread local memory adds 100–600 cycle reload latency;
  // combined with low occupancy the warp scheduler has no other work to hide it.
  // Fix: reduce register pressure (loop unrolling, variable scoping, __launch_bounds__).
  if (pattern.spill_risk && occupancy < 0.4) {
    matched.push({
      tag: "register_spill",
      suggestion: "Reduce registers; add __launch_bounds__",
    });
  }

  // Rule 2 — Latency-bound memory (E1)
  // Cross: memory.class === "memory_bound" × pattern.stall_memory_dependency.
  // Back-to-back loads with no intervening compute (stream_max_consecutive_loads > 4,
  // interleave_score < 0.2) stall the warp on each response before the next request
  // can be issued.  Fix: software prefetch / pipeline loads ahead of dependent computes.
  if (
    memory.class === "memory_bound" &&
    pattern.stall_memory_dependency
  ) {
    matched.push({
      tag: "memory_latency",
      suggestion: "Prefetch; software pipelining",
    });
  }

  // Rule 3 — Bandwidth-saturating memory (E2)
  // Cross: memory.class === "memory_bound" × pattern.stall_memory_throttle.
  // Good load/compute interleaving (score > 0.4) but more memory than compute
  // (computeToMemory < 2.0) means the L2/DRAM pipeline stays perpetually full.
  // Fix: vectorize to LDG.128; eliminate redundant global traffic; use caching.
  if (
    memory.class === "memory_bound" &&
    pattern.stall_memory_throttle
  ) {
    matched.push({
      tag: "memory_bandwidth",
      suggestion:
        "Vectorize loads to LDG.128; eliminate redundant traffic",
    });
  }

  // Rule 4 — Atomic contention
  // Cross: pattern.atomic_contention_risk (ATOM/RED > 5% of global ops) ×
  //        pattern.read_modify_write (load ≈ store with low compute).
  // Both signals independently suggest histogram/scatter RMW shapes;
  // together they strongly confirm L2-serialised atomic bottlenecks.
  // Fix: per-warp privatization in shared memory, then one atomic per warp.
  if (pattern.atomic_contention_risk && pattern.read_modify_write) {
    matched.push({
      tag: "atomic_contention",
      suggestion:
        "Privatize; warp-shuffle reduction before atomic",
    });
  }

  // Rule 5 — SFU throughput
  // Cross: pattern.sfu_heavy (MUFU > 15% of arithmetic) × NOT memory_bound.
  // The SFU pipeline runs at 1/4 the FP32 CUDA-core throughput per SM.
  // When the kernel is not memory-bound, SFU stalls are the dominant wall.
  // Fix: replace transcendental MUFU calls with polynomial approximations.
  if (pattern.sfu_heavy && memory.class !== "memory_bound") {
    matched.push({
      tag: "sfu_throughput",
      suggestion: "Replace MUFU with polynomial approx",
    });
  }

  // Rule 6 — Sync overhead (E4)
  // Cross: pattern.stall_sync (barriers ≥ 2, work_per_barrier < 30, shared present) ×
  //        pattern.class === "tiled".
  // Tiled kernels keep threads in lock-step with __syncthreads(); too many barriers
  // per tile iteration drain the warp scheduler.
  // Fix: merge cooperative phases; move sync outside inner loops.
  if (pattern.stall_sync && pattern.class === "tiled") {
    matched.push({
      tag: "sync_overhead",
      suggestion: "Merge tile phases; reduce barriers per loop",
    });
  }

  // Rule 7 — Uncoalesced access (load-side OR store-side)
  // Signals: pattern.uncoalesced_risk  — >75% of typed loads are LDG.32.
  //          pattern.store_uncoalesced_risk — >75% of typed stores are STG.32.
  // Either condition causes warp serialisation: the memory controller issues
  // one sub-transaction per thread instead of one 128-byte coalesced burst.
  // Fix: restructure to SoA layout, transpose accesses, or pad stride to 128 bytes.
  if (pattern.uncoalesced_risk || pattern.store_uncoalesced_risk) {
    matched.push({
      tag: "uncoalesced_access",
      suggestion: "Transpose; SoA layout",
    });
  }

  // Rule 8 — Occupancy limited by registers
  // Cross: occ.limiting_factor === "registers" × occupancy < 0.4.
  // When the register file is the binding constraint AND resulting occupancy is
  // already low, the kernel has little instruction-level parallelism to hide
  // memory latency.  occ.register_pressure_margin (if set) gives the concrete
  // regs/thread reduction needed to reach the next tier (next_occupancy_class).
  // Fix: add __launch_bounds__(threadsPerBlock) or reduce register demand.
  if (occ.limiting_factor === "registers" && occupancy < 0.4) {
    matched.push({
      tag: "occupancy",
      suggestion: "Use __launch_bounds__",
    });
  }

  // Rule 9 — Compute-bound with missed Tensor Core opportunity
  // Cross: memory.class === "compute_friendly" × pattern.missing_tensor_cores.
  // A compute-friendly kernel that does heavy scalar FFMA with zero MMA/HMMA/WGMMA
  // instructions is leaving 4–16× throughput on the table on SM ≥ 7.0 (Volta+).
  // Fix: switch to a Tensor Core API (cuBLAS, cuDNN, CUTLASS, wmma::).
  if (
    memory.class === "compute_friendly" &&
    pattern.missing_tensor_cores
  ) {
    matched.push({
      tag: "compute_bound",
      suggestion: "Switch to Tensor Core API",
    });
  }

  // ── Assemble result ────────────────────────────────────────────────────────

  // Default: "none_detected" when no rule fires (no dominant bottleneck found).
  const primaryRule = matched[0] ?? {
    tag: "none_detected",
    suggestion: "No dominant bottleneck detected",
  };

  // stall_profile mirrors the four Group E stall signals from the pattern model.
  // local_memory adds an occupancy cross-check (< 0.5): even heavy spill I/O
  // is only a scheduling stall when occupancy is low enough that the scheduler
  // cannot hide it with other ready warps.
  const stall_profile = {
    memory_dependency: pattern.stall_memory_dependency,
    memory_throttle: pattern.stall_memory_throttle,
    local_memory: pattern.stall_local_memory && occupancy < 0.5,
    sync: pattern.stall_sync,
  };

  // Confidence: fraction of the 9 rules that fired, clamped to [0, 1].
  // 1 rule fired  ⇒ 0.33  (weak signal)
  // 3 rules fired ⇒ 1.00  (strong multi-model agreement)
  const confidence = Math.min(matched.length / 3, 1.0);

  return {
    primary_bottleneck: primaryRule.tag,
    secondary_bottlenecks: matched.slice(1).map((r) => r.tag),
    stall_profile,
    optimization_priority: matched.map((r) => r.suggestion),
    confidence: Math.round(confidence * 100) / 100,
  };
}
