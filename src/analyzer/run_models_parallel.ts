/**
 * Mirrors pipeline.py ThreadPoolExecutor over occupancy_model, memory, pattern.

 * Uses three Node worker threads so work can run on multiple cores (Python threads
 * are limited by the GIL for CPU-bound pure Python; Node workers are separate isolates).
 */

import { runWorkerScript } from "./worker_util";
import {
  analyzeOccupancyModel,
  type OccupancyModelResult,
} from "./occupancy_model";
import { analyzeMemory, type MemoryAnalysis } from "./memory_model";
import { analyzePattern, type PatternResult } from "./pattern_model";
import type { GpuSpec } from "./gpu_spec";
import type { PtxInstructionFeatures } from "./ptx_features";
import type { SassInstructionFeatures } from "./sass_features";

export interface ParallelModelsInput {
  threads: number;
  shared: number;
  registers: number;
  spec: GpuSpec;
  threadsSource: string;
  sharedSource: string;
  registerSource: string;
  instr: PtxInstructionFeatures;
  sassInstr: SassInstructionFeatures | undefined;
}

function runModelsSync(input: ParallelModelsInput): {
  occModel: OccupancyModelResult;
  memory: MemoryAnalysis;
  pattern: PatternResult;
} {
  return {
    occModel: analyzeOccupancyModel(
      input.threads,
      input.shared,
      input.registers,
      input.spec,
      input.threadsSource,
      input.sharedSource,
      input.registerSource
    ),
    memory: analyzeMemory(input.instr, input.sassInstr),
    pattern: analyzePattern(input.instr, input.sassInstr),
  };
}

/** Three workers (occ / memory / pattern); parent `Promise.all` gathers results. Sync fallback only if workers fail. */
export async function runModelsParallelOrSync(
  input: ParallelModelsInput
): Promise<{
  occModel: OccupancyModelResult;
  memory: MemoryAnalysis;
  pattern: PatternResult;
}> {
  const sass = input.sassInstr ?? null;

  try {
    const [occModel, memory, pattern] = await Promise.all([
      runWorkerScript("workers/occWorker.js", {
        threads: input.threads,
        shared: input.shared,
        registers: input.registers,
        spec: input.spec,
        threadsSource: input.threadsSource,
        sharedSource: input.sharedSource,
        registerSource: input.registerSource,
      }) as Promise<OccupancyModelResult>,
      runWorkerScript("workers/memWorker.js", {
        instr: input.instr,
        sassInstr: sass,
      }) as Promise<MemoryAnalysis>,
      runWorkerScript("workers/patWorker.js", {
        instr: input.instr,
        sassInstr: sass,
      }) as Promise<PatternResult>,
    ]);
    return { occModel, memory, pattern };
  } catch {
    return runModelsSync(input);
  }
}
