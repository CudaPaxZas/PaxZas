import type { GpuSpec } from "./gpu_spec";
import { AMPERE_LIKE_DEFAULT } from "./gpu_spec";
import {
  computeOccupancyBreakdown,
  type LimitName,
} from "./occupancy_model";

export interface SweepPoint {
  blockSize: number;
  occupancy: number;
  blocksPerSm: number;
  limitingFactor: LimitName;
  blocksByThreads: number;
  blocksByWarps: number;
  blocksByShared: number;
  blocksByRegisters: number;
  blocksByBlockLimit: number;
}

/** User-facing signal badge distilled from pattern + memory + diagnosis. */
export interface KernelSignal {
  id: string;
  label: string;
  severity: "info" | "warn" | "error";
}

/**
 * Kernel-level insights computed once (independent of block size).
 * Attached to SweepResult so the UI can show contextual explanation
 * alongside the sweep without recomputing per point.
 */
export interface KernelInsights {
  memoryClass: string;
  memoryInsight: string;
  memoryConfidence: number;
  arithmeticIntensity: number | null;
  cachePolicy: string | null;
  reuseRatio: number;
  memComputeRatio: number;
  loadStoreRatio: number;
  patternClass: string;
  patternInsight: string;
  patternConfidence: number;
  archetype: string | undefined;
  archetypeDescriptors: string[];
  primaryBottleneck: string;
  secondaryBottlenecks: string[];
  suggestions: string[];
  signals: KernelSignal[];
  occupancyConfidence: number;
  diagnosisConfidence: number;
  stallProfile: {
    memoryDependency: boolean;
    memoryThrottle: boolean;
    localMemory: boolean;
    sync: boolean;
  };
  launchWarnings: string[];
  sources: {
    threads: string;
    shared: string;
    registers: string;
  };
  registerPressureMargin: number | undefined;
  nextOccupancyClass: string | undefined;
  estimatedSmUtilization: string | undefined;
  wasteMetrics:
    | {
        unusedThreadsPerSm: number;
        unusedWarpsPerSm: number;
        unusedRegistersPerSm: number;
        unusedSharedMemBytesPerSm: number;
      }
    | undefined;
  vectorizationScore: number;
  storeVectorizationScore: number;
  storeUncoalescedRisk: boolean;
  ptxOnly: boolean;
}

export interface SweepResult {
  gpu: string;
  sharedMemPerBlock: number;
  registersPerThread: number;
  currentBlockSize: number | undefined;
  points: SweepPoint[];
  insights: KernelInsights | undefined;
}

/**
 * Build the 5 user-facing signals from raw model outputs.
 * Maps internal boolean flags to a short, prioritised badge list.
 */
export function buildSignals(
  pattern: {
    spill_risk: boolean;
    uncoalesced_risk: boolean;
    store_uncoalesced_risk: boolean;
    atomic_contention_risk: boolean;
    stall_sync: boolean;
    missing_tensor_cores: boolean;
    sfu_heavy: boolean;
    uses_tensor_cores: boolean;
    vectorization_score: number;
  },
  memory: {
    class: string;
    store_vectorization_score: number;
  },
): KernelSignal[] {
  const out: KernelSignal[] = [];
  if (pattern.spill_risk) {
    out.push({ id: "spill", label: "Register spill", severity: "error" });
  }
  if (pattern.uncoalesced_risk) {
    out.push({ id: "uncoalesced", label: "Poor memory access", severity: "error" });
  }
  if (pattern.store_uncoalesced_risk) {
    out.push({ id: "store_uncoalesced", label: "Poor store coalescing", severity: "error" });
  }
  if (pattern.atomic_contention_risk) {
    out.push({ id: "atomic", label: "Atomic contention", severity: "warn" });
  }
  if (pattern.stall_sync) {
    out.push({ id: "sync", label: "Sync overhead", severity: "warn" });
  }
  if (pattern.uses_tensor_cores) {
    out.push({ id: "tensor", label: "Tensor cores", severity: "info" });
  } else if (pattern.missing_tensor_cores) {
    out.push({ id: "tensor_miss", label: "Missing tensor cores", severity: "warn" });
  }
  if (pattern.sfu_heavy && memory.class !== "memory_bound") {
    out.push({ id: "sfu", label: "SFU heavy", severity: "warn" });
  }
  out.push({
    id: "vec_ldg",
    label: `LDG vec ${pattern.vectorization_score.toFixed(2)}`,
    severity: pattern.vectorization_score < 0.5 ? "warn" : "info",
  });
  out.push({
    id: "vec_stg",
    label: `STG vec ${memory.store_vectorization_score.toFixed(2)}`,
    severity: memory.store_vectorization_score < 0.5 ? "warn" : "info",
  });
  return out;
}

/**
 * Sweep block sizes from `warpSize` to `smMaxThreads` (stepping by warpSize)
 * and return occupancy metrics at each point.  Runs synchronously since
 * `computeOccupancyBreakdown` is pure arithmetic.
 */
export function sweepBlockSizes(
  sharedMemPerBlock: number,
  registersPerThread: number,
  spec: GpuSpec = AMPERE_LIKE_DEFAULT,
  currentBlockSize?: number,
  insights?: KernelInsights,
): SweepResult {
  const step = spec.warpSize;
  const maxThreads = spec.smMaxThreads;
  const points: SweepPoint[] = [];

  for (let bs = step; bs <= maxThreads; bs += step) {
    const bd = computeOccupancyBreakdown(bs, sharedMemPerBlock, registersPerThread, spec);
    points.push({
      blockSize: bs,
      occupancy: bd.occupancy,
      blocksPerSm: bd.blocks_per_sm,
      limitingFactor: bd.limiting_factor,
      blocksByThreads: bd.blocks_by_threads,
      blocksByWarps: bd.blocks_by_warps,
      blocksByShared: bd.blocks_by_shared,
      blocksByRegisters: bd.blocks_by_registers,
      blocksByBlockLimit: bd.blocks_by_block_limit,
    });
  }

  return {
    gpu: spec.name,
    sharedMemPerBlock,
    registersPerThread,
    currentBlockSize,
    points,
    insights,
  };
}
