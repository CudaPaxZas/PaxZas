/**
 * Memory intensity model (matches memory_model.py).
 */

import type { PtxInstructionFeatures } from "./ptx_features";
import { bytesHeuristic, flopsHeuristic } from "./ptx_features";
import type { SassInstructionFeatures } from "./sass_features";

function safeDiv(n: number, d: number): number {
  if (d === 0) {
    return 0;
  }
  return n / d;
}

function estimateSassBytes(
  sass: SassInstructionFeatures,
  defaultBytesPerOp: number
): number {
  const knownLdg = sass.ldg_128 + sass.ldg_64 + sass.ldg_32;
  const unknownLdg = Math.max(0, sass.global_loads - knownLdg);
  return (
    sass.ldg_128 * 16 +
    sass.ldg_64 * 8 +
    sass.ldg_32 * 4 +
    unknownLdg * defaultBytesPerOp +
    sass.global_stores * defaultBytesPerOp
  );
}

export interface MemoryAnalysis {
  class: string;
  global_loads: number;
  global_stores: number;
  shared_loads: number;
  shared_stores: number;
  global_mem_ops: number;
  shared_mem_ops: number;
  memory_pressure: number;
  global_mem_source: string;
  shared_mem_source: string;
  compute_source: string;
  cache_policy: string | null;
  confidence: number;
  arithmetic_intensity_ops_per_byte: number | null;
  reuse_ratio: number;
  reuse_strength: number;
  mem_compute_ratio: number;
  bytes_proxy: number;
  flops_proxy: number;
  ptx_flops_proxy: number;
  sass_flops_proxy: number | null;
  ptx_bytes_proxy: number;
  insight: string;
}

export function analyzeMemory(
  ptxFeatures: PtxInstructionFeatures,
  sassFeatures: SassInstructionFeatures | undefined,
  bytesPerMemOp = 4
): MemoryAnalysis {
  const ptxGlobalLoads = ptxFeatures.global_loads;
  const ptxGlobalStores = ptxFeatures.global_stores;
  const ptxGlobalMemOps = ptxGlobalLoads + ptxGlobalStores;
  const ptxFlops = flopsHeuristic(ptxFeatures);

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
    sassFlopsProxy =
      sassFeatures.arithmetic_ops + 2 * sassFeatures.tensor_ops;
    sassBytesProxy = estimateSassBytes(sassFeatures, bytesPerMemOp);
  }

  const hasSassGlobal = sassGlobalLoads + sassGlobalStores > 0;
  const hasSassShared = sassSharedLoads + sassSharedStores > 0;
  const globalLoads = hasSassGlobal ? sassGlobalLoads : ptxGlobalLoads;
  const globalStores = hasSassGlobal ? sassGlobalStores : ptxGlobalStores;
  const sharedLoads = hasSassShared ? sassSharedLoads : 0;
  const sharedStores = hasSassShared ? sassSharedStores : 0;

  const globalMemOps = globalLoads + globalStores;
  const sharedMemOps = sharedLoads + sharedStores;
  const flops = Math.max(sassFlopsProxy, ptxFlops);
  const bytesMoved = hasSassGlobal ? sassBytesProxy : globalMemOps * bytesPerMemOp;

  const memoryPressure = bytesMoved;

  let arithmeticIntensity: number;
  if (bytesMoved <= 0) {
    arithmeticIntensity = flops > 0 ? Infinity : 0;
  } else {
    arithmeticIntensity = flops / bytesMoved;
  }

  const reuseRatio = safeDiv(sharedLoads, globalLoads);
  const reuseStrength = safeDiv(sharedLoads, globalMemOps);
  const memComputeRatio = safeDiv(bytesMoved, flops);

  let cachePolicy: string | null = null;
  if (sassFeatures !== undefined) {
    if (sassFeatures.cg_loads > 0) {
      cachePolicy = "L2";
    } else if (sassFeatures.cs_loads > 0) {
      cachePolicy = "streaming";
    }
  }
  const isStreaming = cachePolicy === "streaming";

  let confidence = 0.6;
  if (sassFeatures !== undefined) {
    confidence += 0.25;
  }
  if (globalMemOps === 0) {
    confidence = 0.3;
  } else {
    if (
      reuseRatio > 2.0 ||
      (arithmeticIntensity !== Infinity && arithmeticIntensity < 0.5)
    ) {
      confidence += 0.1;
    }
  }
  confidence = Math.min(0.99, confidence);

  const highPressureThresholdBytes = 2048;
  let memClass: string;
  let insight: string;

  if (arithmeticIntensity === Infinity) {
    memClass = "compute_friendly";
    insight = "no global bytes proxy with nonzero compute proxy";
  } else if (isStreaming && arithmeticIntensity < 0.5) {
    memClass = "memory_bound";
    insight = "streaming access pattern with low reuse";
  } else if (arithmeticIntensity < 0.5 && bytesMoved > highPressureThresholdBytes) {
    memClass = "memory_bound";
    insight = "high bandwidth pressure";
  } else if (reuseRatio > 2.0 && reuseStrength > 1.0) {
    memClass = "reuse_optimized";
    insight = "shared-memory traffic indicates reuse-oriented behavior";
  } else if (arithmeticIntensity > 2.0) {
    memClass = "compute_friendly";
    insight = "compute-heavy intensity with moderate memory pressure";
  } else {
    memClass = "balanced";
    insight = "memory and compute proxies are not strongly imbalanced";
  }

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
    compute_source: sassFlopsProxy > 0 ? "sass" : "ptx",
    cache_policy: cachePolicy,
    confidence: Math.round(confidence * 100) / 100,
    arithmetic_intensity_ops_per_byte:
      arithmeticIntensity === Infinity
        ? null
        : Math.round(arithmeticIntensity * 1e6) / 1e6,
    reuse_ratio: Math.round(reuseRatio * 1e6) / 1e6,
    reuse_strength: Math.round(reuseStrength * 1e6) / 1e6,
    mem_compute_ratio: Math.round(memComputeRatio * 1e6) / 1e6,
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
