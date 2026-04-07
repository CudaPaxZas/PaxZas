/**
 * Pattern classification (matches pattern_model.py).
 */

import type { PtxInstructionFeatures } from "./ptx_features";
import type { SassInstructionFeatures } from "./sass_features";
import { safeDiv } from "./opcode_kinds";

export interface PatternMicro {
  high_looping: boolean;
  sync_heavy: boolean;
  control_irregular: boolean;
  control_dominated: boolean;
  complex_kernel: boolean;
  tensor_dominated: boolean;
  streaming: boolean;
  sync_efficiency: string;
  interleaving: string;
}

export interface PatternResult {
  class: string;
  confidence: number;
  shared_ops: number;
  global_ops: number;
  barriers: number;
  branches: number;
  loops: number;
  shared_to_global_ratio: number;
  branch_density: number;
  branch_per_global_mem_op: number;
  barrier_density: number;
  work_per_barrier: number;
  compute_to_memory_ratio: number;
  uses_tensor_cores: boolean;
  streaming: boolean;
  sync_efficiency: string;
  interleaving: string;
  pattern_micro: PatternMicro;
  source: string;
  insight: string;
}

export function analyzePattern(
  ptxFeatures: PtxInstructionFeatures,
  sassFeatures: SassInstructionFeatures | undefined
): PatternResult {
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

  const ptxShared = 0;
  const ptxBarriers = ptxFeatures.barrier;
  const ptxBranches = ptxFeatures.branches;
  const ptxLoops = ptxFeatures.loops;

  const sharedOps = sassShared > 0 ? sassShared : ptxShared;
  const barriers = sassBarriers > 0 ? sassBarriers : ptxBarriers;
  const branches = sassBranches > 0 ? sassBranches : ptxBranches;
  const loops = ptxLoops;

  const globalOps =
    sassGlobal > 0
      ? sassGlobal
      : ptxFeatures.global_loads + ptxFeatures.global_stores;

  const computeOps =
    sassCompute > 0 ? sassCompute : ptxFeatures.fma + ptxFeatures.add + ptxFeatures.mul;

  const usesTensor =
    sassFeatures !== undefined && sassFeatures.tensor_ops > 0;

  const sharedToGlobal = safeDiv(sharedOps, globalOps);
  const branchDensity = safeDiv(branches, computeOps + 1);
  const branchPerMem = safeDiv(branches, globalOps + 1);
  const computeToMemory = safeDiv(computeOps, globalOps + 1);

  const hasShared = sharedOps > 0;
  const hasBarrier = barriers > 0;
  const hasLoops = loops > 0;
  const highBranching =
    branchDensity > 0.1 ||
    (branchPerMem > 0.08 && computeToMemory < 4.0);

  const loopDensity = safeDiv(loops, computeOps + 1);
  const barrierDensity = safeDiv(barriers, computeOps + 1);
  const controlIntensity = branchDensity + branchPerMem;
  const computeVsControl = safeDiv(computeOps, branches + 1);
  const structuralComplexity =
    (hasLoops ? 1 : 0) + (hasBarrier ? 1 : 0) + (highBranching ? 1 : 0);
  const tensorDominated = usesTensor && computeToMemory > 4.0;

  const highLooping = loopDensity > 0.05;
  const syncHeavy = barrierDensity > 0.02;
  const controlIrregular = controlIntensity > 0.15;
  const controlDominated = computeVsControl < 5.0;
  const complexKernel = structuralComplexity >= 2;
  const isStreaming =
    !hasShared && !hasBarrier && !hasLoops && computeToMemory < 2.0;
  const workPerBarrier = safeDiv(computeOps, barriers + 1);
  let syncEfficiency: string;
  if (workPerBarrier > 100) {
    syncEfficiency = "efficient";
  } else if (workPerBarrier > 30) {
    syncEfficiency = "moderate";
  } else {
    syncEfficiency = "inefficient";
  }

  const interleaveScore =
    sassFeatures !== undefined ? sassFeatures.stream_interleave_score : 0;
  const maxLoadRun =
    sassFeatures !== undefined
      ? sassFeatures.stream_max_consecutive_loads
      : 0;
  let interleavingPattern: string;
  if (interleaveScore > 0.4) {
    interleavingPattern = "interleaved";
  } else if (maxLoadRun > 4) {
    interleavingPattern = "stacked";
  } else {
    interleavingPattern = "mixed";
  }

  const patternMicro: PatternMicro = {
    high_looping: highLooping,
    sync_heavy: syncHeavy,
    control_irregular: controlIrregular,
    control_dominated: controlDominated,
    complex_kernel: complexKernel,
    tensor_dominated: tensorDominated,
    streaming: isStreaming,
    sync_efficiency: syncEfficiency,
    interleaving: interleavingPattern,
  };

  let pattern: string;
  let confidence: number;
  let insight: string;

  if (highBranching) {
    pattern = "control_heavy";
    confidence = 0.85;
    insight =
      "high branching (vs compute or memory) suggests control-flow dominated kernel";
  } else if (hasShared && hasBarrier && hasLoops) {
    if (computeToMemory > 8) {
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
      pattern = "reduction";
      confidence = 0.82;
      insight =
        "shared memory, barriers, and loops with modest compute/memory ratio " +
        "suggest reduction";
    }
  } else if (!hasShared && !hasBarrier) {
    if (computeToMemory < 2) {
      pattern = "elementwise";
      confidence = 0.8;
      insight =
        "low compute vs global traffic without shared/sync suggests elementwise work";
    } else {
      pattern = "compute_heavy";
      confidence = 0.78;
      insight =
        "high compute vs global ops without shared/sync suggests compute-bound kernel";
    }
  } else {
    pattern = "irregular";
    confidence = 0.6;
    insight = "mixed signals indicate irregular or data-dependent behavior";
  }

  if (sassFeatures !== undefined) {
    confidence += 0.1;
  }
  confidence = Math.min(confidence, 0.95);

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
    pattern_micro: patternMicro,
    source: sassFeatures !== undefined ? "sass" : "ptx",
    insight,
  };
}
