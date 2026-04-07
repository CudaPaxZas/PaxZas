/**
 * Bottleneck heuristic (matches bottleneck.py).
 */

import type { PtxInstructionFeatures } from "./ptx_features";
import { bytesHeuristic, flopsHeuristic } from "./ptx_features";

export type BottleneckLabel = "memory-bound" | "compute-bound" | "balanced" | "unknown";

export interface BottleneckResult {
  bottleneck: BottleneckLabel;
  arithmetic_intensity_ops_per_byte: number | null;
  flops_proxy: number;
  global_mem_ops: number;
  notes: string[];
}

export function heuristicBottleneck(
  features: PtxInstructionFeatures,
  registersPerThread: number,
  intensityThreshold = 8.0,
  highRegPressure = 64
): BottleneckResult {
  const flops = flopsHeuristic(features);
  const memOps = features.global_loads + features.global_stores;
  const bytesMoved = bytesHeuristic(features);
  let bottleneck: BottleneckLabel;
  let intensity: number;

  if (bytesMoved <= 0 && flops <= 0) {
    bottleneck = "unknown";
    intensity = 0;
  } else if (bytesMoved <= 0) {
    bottleneck = "compute-bound";
    intensity = Infinity;
  } else {
    intensity = flops / bytesMoved;
    if (intensity < intensityThreshold / 4) {
      bottleneck = "memory-bound";
    } else if (intensity > intensityThreshold) {
      bottleneck = "compute-bound";
    } else {
      bottleneck = "balanced";
    }
  }

  const notes: string[] = [];
  if (registersPerThread >= highRegPressure) {
    notes.push("high_register_pressure");
  }
  if (features.fma > memOps && memOps > 0) {
    notes.push("fma_heavy_vs_global_mem_ops");
  }
  if (memOps > 0 && flops < memOps) {
    notes.push("low_flops_per_global_access");
  }

  return {
    bottleneck,
    arithmetic_intensity_ops_per_byte:
      intensity === Infinity ? null : Math.round(intensity * 1e6) / 1e6,
    flops_proxy: Math.round(flops * 1000) / 1000,
    global_mem_ops: memOps,
    notes,
  };
}
