/**
 * Tests for src/analyzer/diagnose.ts — cross-model diagnosis (Group G).
 *
 * Strategy: construct minimal MemoryAnalysis / KernelAnalysis / PatternResult
 * stubs that satisfy exactly the signals under test, then assert the output.
 * Using analyzeKernel() for KernelAnalysis to avoid duplicating that logic.
 */
import { describe, expect, it } from "vitest";
import { diagnoseKernel } from "../src/analyzer/diagnose";
import { analyzeKernel } from "../src/analyzer/occupancy_model";
import type { MemoryAnalysis } from "../src/analyzer/memory_model";
import type { PatternResult } from "../src/analyzer/pattern_model";

// ── Stub helpers ──────────────────────────────────────────────────────────────

/** Returns a baseline MemoryAnalysis with the given class and all optional fields zeroed. */
function memStub(cls: string): MemoryAnalysis {
  return {
    class: cls,
    global_loads: 0,
    global_stores: 0,
    shared_loads: 0,
    shared_stores: 0,
    global_mem_ops: 0,
    shared_mem_ops: 0,
    memory_pressure: 0,
    global_mem_source: "ptx",
    shared_mem_source: "unknown",
    compute_source: "ptx",
    cache_policy: null,
    confidence: 1,
    arithmetic_intensity_ops_per_byte: null,
  } as unknown as MemoryAnalysis;
}

/** Clean PatternResult with every boolean false and metrics zeroed. */
function patStub(overrides: Partial<PatternResult> = {}): PatternResult {
  const base: PatternResult = {
    class: "elementwise",
    confidence: 1,
    shared_ops: 0,
    global_ops: 4,
    barriers: 0,
    branches: 0,
    loops: 0,
    shared_to_global_ratio: 0,
    branch_density: 0,
    branch_per_global_mem_op: 0,
    barrier_density: 0,
    work_per_barrier: 0,
    compute_to_memory_ratio: 2,
    uses_tensor_cores: false,
    high_looping: false,
    sync_heavy: false,
    control_irregular: false,
    control_dominated: false,
    complex_kernel: false,
    tensor_dominated: false,
    streaming: false,
    sync_efficiency: "none",
    interleaving: "none",
    spill_risk: false,
    uncoalesced_risk: false,
    missing_tensor_cores: false,
    atomic_contention_risk: false,
    sfu_heavy: false,
    vectorization_score: 1,
    over_synchronized: false,
    fp16_scalar_risk: false,
    read_modify_write: false,
    stall_memory_dependency: false,
    stall_memory_throttle: false,
    stall_local_memory: false,
    stall_sync: false,
    uses_warp_shuffle: false,
    uses_warp_vote: false,
    warp_reduction_pattern: false,
    archetype: undefined,
    source: "ptx",
    insight: "",
  };
  return { ...base, ...overrides };
}

/** KernelAnalysis for a moderately-occupied kernel (threads=256, shared=0, regs=32). */
function occStub(occupancyOverride?: number) {
  const k = analyzeKernel(256, 0, 32);
  if (occupancyOverride !== undefined) {
    // KernelAnalysis extends Record<string, unknown> so we can override.
    return { ...k, occupancy: occupancyOverride };
  }
  return k;
}

// ── Rule 1: register_spill ────────────────────────────────────────────────────

describe("diagnoseKernel", () => {
  it("rule1_register_spill_when_spill_risk_and_low_occupancy", () => {
    const mem = memStub("balanced");
    const occ = occStub(0.25); // < 0.4
    const pat = patStub({ spill_risk: true });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).toBe("register_spill");
    expect(d.optimization_priority[0]).toContain("__launch_bounds__");
  });

  it("rule1_does_not_fire_when_occupancy_adequate", () => {
    const mem = memStub("balanced");
    const occ = occStub(0.75); // > 0.4
    const pat = patStub({ spill_risk: true });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).not.toBe("register_spill");
  });

  // ── Rule 2: memory_latency ────────────────────────────────────────────────

  it("rule2_memory_latency_when_memory_bound_and_stall_dependency", () => {
    const mem = memStub("memory_bound");
    const occ = occStub(0.6);
    const pat = patStub({ stall_memory_dependency: true });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).toBe("memory_latency");
    expect(d.stall_profile.memory_dependency).toBe(true);
  });

  it("rule2_does_not_fire_for_compute_friendly", () => {
    const mem = memStub("compute_friendly");
    const occ = occStub(0.6);
    const pat = patStub({ stall_memory_dependency: true });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).not.toBe("memory_latency");
  });

  // ── Rule 3: memory_bandwidth ──────────────────────────────────────────────

  it("rule3_memory_bandwidth_when_memory_bound_and_stall_throttle", () => {
    const mem = memStub("memory_bound");
    const occ = occStub(0.6);
    const pat = patStub({ stall_memory_throttle: true });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).toBe("memory_bandwidth");
    expect(d.stall_profile.memory_throttle).toBe(true);
    expect(d.optimization_priority[0]).toContain("LDG.128");
  });

  // ── Rule 4: atomic_contention ─────────────────────────────────────────────

  it("rule4_atomic_contention_when_both_risk_and_rmw", () => {
    const mem = memStub("balanced");
    const occ = occStub(0.6);
    const pat = patStub({ atomic_contention_risk: true, read_modify_write: true });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).toBe("atomic_contention");
    expect(d.optimization_priority[0]).toContain("warp-shuffle");
  });

  // ── Rule 5: sfu_throughput ────────────────────────────────────────────────

  it("rule5_sfu_throughput_when_sfu_heavy_and_not_memory_bound", () => {
    const mem = memStub("compute_friendly");
    const occ = occStub(0.6);
    const pat = patStub({ sfu_heavy: true });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).toBe("sfu_throughput");
    expect(d.optimization_priority[0]).toContain("polynomial");
  });

  it("rule5_does_not_fire_when_memory_bound", () => {
    const mem = memStub("memory_bound");
    const occ = occStub(0.6);
    const pat = patStub({ sfu_heavy: true });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).not.toBe("sfu_throughput");
  });

  // ── Rule 6: sync_overhead ─────────────────────────────────────────────────

  it("rule6_sync_overhead_when_stall_sync_and_tiled", () => {
    const mem = memStub("balanced");
    const occ = occStub(0.6);
    const pat = patStub({ stall_sync: true, class: "tiled" });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).toBe("sync_overhead");
    expect(d.stall_profile.sync).toBe(true);
  });

  it("rule6_does_not_fire_for_non_tiled_pattern", () => {
    const mem = memStub("balanced");
    const occ = occStub(0.6);
    const pat = patStub({ stall_sync: true, class: "elementwise" });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).not.toBe("sync_overhead");
  });

  // ── Rule 7: uncoalesced_access ────────────────────────────────────────────

  it("rule7_uncoalesced_access_when_uncoalesced_risk", () => {
    const mem = memStub("balanced");
    const occ = occStub(0.6);
    const pat = patStub({ uncoalesced_risk: true });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).toBe("uncoalesced_access");
    expect(d.optimization_priority[0]).toContain("Transpose");
  });

  // ── Rule 9: compute_bound ─────────────────────────────────────────────────

  it("rule9_compute_bound_when_compute_friendly_and_missing_tensor_cores", () => {
    const mem = memStub("compute_friendly");
    const occ = occStub(0.6);
    const pat = patStub({ missing_tensor_cores: true });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).toBe("compute_bound");
    expect(d.optimization_priority[0]).toContain("Tensor Core");
  });

  // ── No bottleneck ─────────────────────────────────────────────────────────

  it("returns_none_when_no_rules_fire", () => {
    const mem = memStub("balanced");
    const occ = occStub(0.8);
    const pat = patStub();
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).toBe("none");
    expect(d.secondary_bottlenecks).toHaveLength(0);
    expect(d.confidence).toBe(0);
  });

  // ── Multiple bottlenecks ──────────────────────────────────────────────────

  it("secondary_bottlenecks_populated_when_multiple_rules_fire", () => {
    // Rules 2 (memory_latency) + 7 (uncoalesced_access) both fire.
    const mem = memStub("memory_bound");
    const occ = occStub(0.6);
    const pat = patStub({
      stall_memory_dependency: true,
      uncoalesced_risk: true,
    });
    const d = diagnoseKernel(mem, occ, pat);
    expect(d.primary_bottleneck).toBe("memory_latency");
    expect(d.secondary_bottlenecks).toContain("uncoalesced_access");
  });

  // ── stall_profile: local_memory cross-check ───────────────────────────────

  it("stall_profile_local_memory_requires_both_spill_and_low_occupancy", () => {
    const mem = memStub("balanced");
    // High occupancy: stall_local_memory alone doesn't trigger local_memory stall.
    const occHigh = occStub(0.8);
    const pat = patStub({ stall_local_memory: true });
    const d = diagnoseKernel(mem, occHigh, pat);
    expect(d.stall_profile.local_memory).toBe(false);

    // Low occupancy: cross-check triggers.
    const occLow = occStub(0.3);
    const d2 = diagnoseKernel(mem, occLow, pat);
    expect(d2.stall_profile.local_memory).toBe(true);
  });

  // ── Confidence scaling ────────────────────────────────────────────────────

  it("confidence_is_higher_when_more_rules_fire", () => {
    const mem = memStub("memory_bound");
    const occ = occStub(0.6);
    const patNone  = patStub();
    const patMulti = patStub({
      stall_memory_dependency: true,
      stall_memory_throttle: true,
      uncoalesced_risk: true,
    });
    const dNone  = diagnoseKernel(mem, occ, patNone);
    const dMulti = diagnoseKernel(mem, occ, patMulti);
    expect(dMulti.confidence).toBeGreaterThan(dNone.confidence);
  });
});
