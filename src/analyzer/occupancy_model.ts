/**
 * Occupancy model (matches occupancy_model.py).
 */

import type { GpuSpec } from "./gpu_spec";
import { AMPERE_LIKE_DEFAULT } from "./gpu_spec";

export type LimitName =
  | "threads"
  | "warps"
  | "shared_mem"
  | "registers"
  | "block_limit";

export interface OccupancyBreakdown {
  blocks_by_threads: number;
  blocks_by_warps: number;
  blocks_by_shared: number;
  blocks_by_registers: number;
  blocks_by_block_limit: number;
  blocks_per_sm: number;
  occupancy: number;
  limiting_factor: LimitName;
}

function roundUp(v: number, unit: number): number {
  if (unit <= 0) {
    return v;
  }
  return Math.ceil(v / unit) * unit;
}

function blocksByShared(
  smMaxShared: number,
  sharedPerBlock: number,
  smMaxBlocks: number,
  allocUnit: number,
  minAlloc: number
): number {
  if (sharedPerBlock <= 0) {
    return smMaxBlocks;
  }
  let sharedRounded = roundUp(sharedPerBlock, allocUnit);
  if (minAlloc > 0) {
    sharedRounded = Math.max(sharedRounded, minAlloc);
  }
  return smMaxShared / sharedRounded;
}

function blocksByRegisters(
  smMaxRegisters: number,
  registersPerThread: number,
  threadsPerBlock: number,
  warpSize: number,
  allocUnitPerWarp: number
): number {
  const fullWarps = Math.floor(threadsPerBlock / warpSize);
  const partialThreads = threadsPerBlock % warpSize;

  const regsPerFullWarp = roundUp(
    registersPerThread * warpSize,
    allocUnitPerWarp
  );
  const regsFull = fullWarps * regsPerFullWarp;
  let regsPartial = 0;
  if (partialThreads > 0) {
    regsPartial = roundUp(
      registersPerThread * partialThreads,
      allocUnitPerWarp
    );
  }
  const regsPerBlock = regsFull + regsPartial;
  if (regsPerBlock <= 0) {
    return 0;
  }
  return smMaxRegisters / regsPerBlock;
}

export function computeOccupancyBreakdown(
  threadsPerBlock: number,
  sharedMemPerBlock: number,
  registersPerThread: number,
  spec: GpuSpec = AMPERE_LIKE_DEFAULT
): OccupancyBreakdown {
  if (threadsPerBlock <= 0) {
    throw new Error("threads_per_block must be positive");
  }
  if (registersPerThread <= 0) {
    throw new Error("registers_per_thread must be positive");
  }

  const g = spec;
  const maxWarps = g.smMaxWarps ?? Math.floor(g.smMaxThreads / g.warpSize);

  const blocksByThreads =
    threadsPerBlock > g.smMaxThreads
      ? 0
      : Math.floor(g.smMaxThreads / threadsPerBlock);

  const warpsPerBlock = Math.ceil(threadsPerBlock / g.warpSize);
  const blocksByWarps =
    warpsPerBlock > 0 ? Math.floor(maxWarps / warpsPerBlock) : 0;

  const blocksBySharedMem = Math.floor(
    blocksByShared(
      g.smMaxSharedMem,
      sharedMemPerBlock,
      g.smMaxBlocks,
      g.sharedMemAllocUnit,
      g.minSharedPerBlockAlloc
    )
  );

  const blocksByRegs = Math.floor(
    blocksByRegisters(
      g.smMaxRegisters,
      registersPerThread,
      threadsPerBlock,
      g.warpSize,
      g.regAllocUnitPerWarp
    )
  );

  const blocksByBlockLimit = g.smMaxBlocks;

  const limits: Record<LimitName, number> = {
    threads: blocksByThreads,
    warps: blocksByWarps,
    shared_mem: blocksBySharedMem,
    registers: blocksByRegs,
    block_limit: blocksByBlockLimit,
  };

  const order: LimitName[] = [
    "threads",
    "warps",
    "shared_mem",
    "registers",
    "block_limit",
  ];
  let limitingFactor: LimitName = order[0]!;
  let minVal = limits[limitingFactor]!;
  for (const k of order) {
    if (limits[k]! < minVal) {
      minVal = limits[k]!;
      limitingFactor = k;
    }
  }
  const blocksPerSm = minVal;

  let occ = (blocksPerSm * threadsPerBlock) / g.smMaxThreads;
  occ = Math.max(0, Math.min(1, occ));

  return {
    blocks_by_threads: blocksByThreads,
    blocks_by_warps: blocksByWarps,
    blocks_by_shared: blocksBySharedMem,
    blocks_by_registers: blocksByRegs,
    blocks_by_block_limit: blocksByBlockLimit,
    blocks_per_sm: blocksPerSm,
    occupancy: occ,
    limiting_factor: limitingFactor,
  };
}

export function collectWarnings(
  threadsPerBlock: number,
  sharedMemPerBlock: number,
  registersPerThread: number,
  breakdown: OccupancyBreakdown,
  spec: GpuSpec = AMPERE_LIKE_DEFAULT
): string[] {
  const ws = spec.warpSize;
  const warnings: string[] = [];

  if (threadsPerBlock < 128) {
    warnings.push("Low threads per block -> poor latency hiding");
  }
  if (threadsPerBlock < 64) {
    warnings.push("Very low threads per block (< 64) -> suspicious for occupancy");
  }
  if (threadsPerBlock % ws !== 0) {
    warnings.push("Threads not a multiple of warp size -> wasted lanes");
  }
  const activeWarps = threadsPerBlock / ws;
  if (activeWarps < 4) {
    warnings.push("Too few warps per block -> poor SM utilization");
  }
  if (breakdown.limiting_factor === "registers") {
    warnings.push(
      "Register allocation is limiting occupancy (check for over-allocation)"
    );
  }
  if (registersPerThread > 64) {
    warnings.push("Very high register usage per thread");
  }
  if (breakdown.limiting_factor === "shared_mem") {
    warnings.push("Shared memory per block is limiting occupancy");
  }
  if (breakdown.occupancy < 0.4) {
    warnings.push("Low occupancy -> likely poor latency hiding");
  }

  const seen = new Set<string>();
  const out: string[] = [];
  for (const w of warnings) {
    if (!seen.has(w)) {
      seen.add(w);
      out.push(w);
    }
  }
  return out;
}

export interface KernelAnalysis extends Record<string, unknown> {
  gpu: string;
  threads_per_block: number;
  shared_mem_per_block: number;
  registers_per_thread: number;
  occupancy_threads: number;
  occupancy_warps: number;
  occupancy: number;
  limiting_factor: LimitName;
  blocks_per_sm: number;
  limits: Record<string, number>;
  warp_metrics: Record<string, number>;
  warnings: string[];
}

export function analyzeKernel(
  threadsPerBlock: number,
  sharedMemPerBlock: number,
  registersPerThread: number,
  spec: GpuSpec = AMPERE_LIKE_DEFAULT
): KernelAnalysis {
  const bd = computeOccupancyBreakdown(
    threadsPerBlock,
    sharedMemPerBlock,
    registersPerThread,
    spec
  );
  const warns = collectWarnings(
    threadsPerBlock,
    sharedMemPerBlock,
    registersPerThread,
    bd,
    spec
  );
  const g = spec;
  const warpSize = g.warpSize;
  const maxWarpsPerSm = g.smMaxWarps ?? Math.floor(g.smMaxThreads / warpSize);
  const warpsPerBlock = Math.ceil(threadsPerBlock / warpSize);
  const registersPerWarp = registersPerThread * warpSize;
  const warpsPerSm = Math.min(
    bd.blocks_per_sm * warpsPerBlock,
    maxWarpsPerSm
  );
  const warpOccupancy =
    maxWarpsPerSm > 0 ? warpsPerSm / maxWarpsPerSm : 0;

  return {
    gpu: g.name,
    threads_per_block: threadsPerBlock,
    shared_mem_per_block: sharedMemPerBlock,
    registers_per_thread: registersPerThread,
    occupancy_threads: Math.round(bd.occupancy * 1e6) / 1e6,
    occupancy_warps: Math.round(warpOccupancy * 1e6) / 1e6,
    occupancy: Math.round(warpOccupancy * 1e6) / 1e6,
    limiting_factor: bd.limiting_factor,
    blocks_per_sm: bd.blocks_per_sm,
    limits: {
      threads: bd.blocks_by_threads,
      warps: bd.blocks_by_warps,
      shared_mem: bd.blocks_by_shared,
      registers: bd.blocks_by_registers,
      block_limit: bd.blocks_by_block_limit,
    },
    warp_metrics: {
      warp_size: warpSize,
      warps_per_block: warpsPerBlock,
      registers_per_warp: registersPerWarp,
      warps_per_sm: warpsPerSm,
      max_warps_per_sm: maxWarpsPerSm,
      warp_occupancy: Math.round(warpOccupancy * 1e6) / 1e6,
    },
    warnings: warns,
  };
}

function classifyOccupancy(occ: number): string {
  if (occ < 0.3) {
    return "low";
  }
  if (occ < 0.6) {
    return "medium";
  }
  return "high";
}

function confidenceFromSources(
  threadsSource: string,
  sharedSource: string,
  registerSource: string
): number {
  let c = 0.65;
  if (threadsSource === "launch") {
    c += 0.1;
  }
  if (sharedSource === "launch") {
    c += 0.1;
  }
  if (registerSource === "launch") {
    c += 0.1;
  } else if (registerSource === "ptx.maxnreg") {
    c += 0.05;
  }
  return Math.min(0.85, c);
}

export interface OccupancyModelResult {
  class: string;
  confidence: number;
  occupancy: number;
  limiting_factor: LimitName;
  blocks_per_sm: number;
  threads_per_block: number;
  shared_mem_per_block: number;
  registers_per_thread: number;
  sources: {
    threads: string;
    shared: string;
    registers: string;
  };
  insight: string;
}

export function analyzeOccupancyModel(
  threadsPerBlock: number,
  sharedMemPerBlock: number,
  registersPerThread: number,
  spec: GpuSpec = AMPERE_LIKE_DEFAULT,
  threadsSource = "unknown",
  sharedSource = "unknown",
  registerSource = "unknown"
): OccupancyModelResult {
  const occ = analyzeKernel(
    threadsPerBlock,
    sharedMemPerBlock,
    registersPerThread,
    spec
  );
  const cls = classifyOccupancy(Number(occ.occupancy));

  return {
    class: cls,
    confidence: Math.round(
      confidenceFromSources(threadsSource, sharedSource, registerSource) * 100
    ) / 100,
    occupancy: occ.occupancy as number,
    limiting_factor: occ.limiting_factor,
    blocks_per_sm: occ.blocks_per_sm,
    threads_per_block: threadsPerBlock,
    shared_mem_per_block: sharedMemPerBlock,
    registers_per_thread: registersPerThread,
    sources: {
      threads: threadsSource,
      shared: sharedSource,
      registers: registerSource,
    },
    insight:
      occ.limiting_factor !== "threads"
        ? "resource pressure limits active blocks"
        : "thread capacity is the primary limit",
  };
}
