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
 * Cross-model diagnosis layer.
 *
 * Fuses three independent models (memory, occupancy, pattern) to infer the
 * dominant bottleneck and derive a ranked optimisation plan.  Each rule is
 * evaluated in priority order; the first match becomes `primary_bottleneck`.
 * All subsequent rule matches contribute to `secondary_bottlenecks`.
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
  // Tag + suggestion pairs.  All rules are evaluated; first hit = primary,
  // rest = secondary.

  type Rule = { tag: string; suggestion: string };
  const matched: Rule[] = [];

  // Rule 1 — Register spill (cross: spill_risk × occupancy)
  if (pattern.spill_risk && occupancy < 0.4) {
    matched.push({
      tag: "register_spill",
      suggestion: "Reduce registers; add __launch_bounds__",
    });
  }

  // Rule 2 — Latency-bound memory (cross: memory_bound × stall_memory_dependency)
  if (
    memory.class === "memory_bound" &&
    pattern.stall_memory_dependency
  ) {
    matched.push({
      tag: "memory_latency",
      suggestion: "Prefetch; software pipelining",
    });
  }

  // Rule 3 — Bandwidth-saturating memory (cross: memory_bound × stall_memory_throttle)
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

  // Rule 4 — Atomic contention (cross: atomic_contention_risk × read_modify_write)
  if (pattern.atomic_contention_risk && pattern.read_modify_write) {
    matched.push({
      tag: "atomic_contention",
      suggestion:
        "Privatize; warp-shuffle reduction before atomic",
    });
  }

  // Rule 5 — SFU throughput bound (cross: sfu_heavy × NOT memory_bound)
  if (pattern.sfu_heavy && memory.class !== "memory_bound") {
    matched.push({
      tag: "sfu_throughput",
      suggestion: "Replace MUFU with polynomial approx",
    });
  }

  // Rule 6 — Sync overhead (cross: stall_sync × tiled pattern)
  if (pattern.stall_sync && pattern.class === "tiled") {
    matched.push({
      tag: "sync_overhead",
      suggestion: "Merge tile phases; reduce barriers per loop",
    });
  }

  // Rule 7 — Uncoalesced access
  if (pattern.uncoalesced_risk) {
    matched.push({
      tag: "uncoalesced_access",
      suggestion: "Transpose; SoA layout",
    });
  }

  // Rule 8 — Occupancy limited by registers (cross: occ.limiting_factor × occupancy)
  if (occ.limiting_factor === "registers" && occupancy < 0.4) {
    matched.push({
      tag: "occupancy",
      suggestion: "Use __launch_bounds__",
    });
  }

  // Rule 9 — Compute bound with missed Tensor Cores
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

  const primaryRule = matched[0] ?? {
    tag: "none",
    suggestion: "No dominant bottleneck detected",
  };

  const stall_profile = {
    memory_dependency: pattern.stall_memory_dependency,
    memory_throttle: pattern.stall_memory_throttle,
    local_memory: pattern.stall_local_memory && occupancy < 0.5,
    sync: pattern.stall_sync,
  };

  // Confidence: fraction of the 9 rules that fired (clamped to [0,1])
  const confidence = Math.min(matched.length / 3, 1.0);

  return {
    primary_bottleneck: primaryRule.tag,
    secondary_bottlenecks: matched.slice(1).map((r) => r.tag),
    stall_profile,
    optimization_priority: matched.map((r) => r.suggestion),
    confidence: Math.round(confidence * 100) / 100,
  };
}
