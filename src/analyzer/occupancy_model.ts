/**
 * Occupancy model (matches occupancy_model.py).
 *
 * This module computes GPU kernel occupancy, which measures how effectively a GPU SM
 * (streaming multiprocessor) is utilized.
 *
 * Occupancy = (active warps on SM) / (max warps on SM)
 *   Range: 0.0 (no threads) to 1.0 (SM fully utilized)
 *   Higher occupancy = better latency hiding and resource utilization
 *
 * Four resource limits can constrain occupancy per kernel launch:
 * 1. Max threads per SM: limits blocks by available thread slots
 * 2. Max warps per SM: limits blocks by warp-level concurrency
 * 3. Shared memory per SM: limits blocks by memory allocation
 * 4. Register file per SM: limits blocks by register allocation
 * 5. Block limit per SM: CUDA hard limit on block count
 *
 * The "limiting factor" identifies which resource is the bottleneck.
 * Understanding this helps optimize kernel configuration:
 * - threads: increase block size
 * - registers: reduce per-thread register usage or block size
 * - shared_mem: reduce per-block shared memory allocation
 * - warps/block_limit: GPU architecture constraint
 *
 * Algorithm:
 * 1. Calculate how many blocks can fit per SM under each limit
 * 2. The minimum becomes the block count limit
 * 3. Occupancy = (blocks * threads) / max_threads_per_sm
 */

import type { GpuSpec } from "./gpu_spec";
import { AMPERE_LIKE_DEFAULT } from "./gpu_spec";

/**
 * LimitName identifies which resource constrains occupancy for a given kernel config.
 * Used to guide optimization strategy.
 */
export type LimitName =
  | "threads"       // Block size exceeds max threads per SM
  | "warps"         // Block warps exceed max warps per SM
  | "shared_mem"    // Per-block shared memory exceeds available
  | "registers"     // Per-block register usage exceeds available
  | "block_limit";  // Hard GPU limit on concurrent blocks

/**
 * Occupancy breakdown shows how many blocks can concurrently fit per SM
 * under each individual constraint.
 *
 * The "limiting_factor" is whichever constraint has the lowest block count.
 * occupancy = (blocks_per_sm * threads_per_block) / smMaxThreads
 */
export interface OccupancyBreakdown {
  blocks_by_threads: number;    // Max blocks limited by thread count
  blocks_by_warps: number;      // Max blocks limited by warp count
  blocks_by_shared: number;     // Max blocks limited by shared memory
  blocks_by_registers: number;  // Max blocks limited by register file
  blocks_by_block_limit: number; // Hard architecture block limit per SM
  blocks_per_sm: number;        // Actual blocks = minimum of above
  occupancy: number;            // (blocks * threads) / max_threads [0.0-1.0]
  limiting_factor: LimitName;   // Which limit is the bottleneck
}

/**
 * Rounds a value up to the nearest multiple of a unit (alignment).
 * Used to account for memory allocation granularity.
 * Example: roundUp(150, 256) = 256 (CUDA allocates in 256-byte chunks)
 *
 * @param v Value to round up
 * @param unit Alignment granule (e.g. 256 for 256-byte chunks)
 * @returns Rounded up value (or v if unit <= 0)
 */
function roundUp(v: number, unit: number): number {
  if (unit <= 0) {
    return v;
  }
  return Math.ceil(v / unit) * unit;
}

/**
 * Calculates maximum blocks limited by shared memory availability.
 *
 * Strategy:
 * 1. If kernel uses no shared memory, return max blocks (no constraint)
 * 2. Round up kernel's shared memory request to allocation unit granularity
 * 3. Enforce minimum allocation (some kernels declare shared but don't use it)
 * 4. Divide total shared memory by per-block amount
 *
 * Example (Ampere SM80):
 *   - Total: 164 KB, Granule: 256 bytes, Min: 256 bytes
 *   - Kernel requests 8 KB -> rounded to 8 KB (8 granules)
 *   - Blocks = 164 KB / 8 KB = 20 blocks
 *
 * @param smMaxShared Total shared memory per SM (bytes)
 * @param sharedPerBlock Kernel's shared memory request (bytes)
 * @param smMaxBlocks Hard block limit (fallback if shared memory is plenty)
 * @param allocUnit Allocation granule (e.g. 256 bytes)
 * @param minAlloc Minimum allocation even for zero request
 * @returns Maximum blocks limited by shared memory
 */
function blocksByShared(
  smMaxShared: number,
  sharedPerBlock: number,
  smMaxBlocks: number,
  allocUnit: number,
  minAlloc: number
): number {
  if (sharedPerBlock <= 0) {
    return smMaxBlocks;  // No shared memory constraint
  }
  let sharedRounded = roundUp(sharedPerBlock, allocUnit);
  if (minAlloc > 0) {
    sharedRounded = Math.max(sharedRounded, minAlloc);
  }
  return smMaxShared / sharedRounded;
}

/**
 * Calculates maximum blocks limited by register file availability.
 *
 * Gap 7 fix: registers are allocated per WARP, not per block.
 * CUDA's register allocator works at warp granularity:
 *   1. Round per-warp demand to regAllocUnitPerWarp (256 on all modern GPUs).
 *   2. Per-block total = warps_per_block × rounded_per_warp_demand.
 *
 * The old per-block granularity (`roundUp(regsPerThread × threads, perBlockUnit)`)
 * overstated occupancy for certain register counts.  Example where they differ:
 *   6 regs/thread, 128 threads (4 warps), Ampere (unit=256):
 *     OLD: roundUp(6×128=768, 256) = 768  → floor(65536/768) = 85 blocks
 *     NEW: 4 × roundUp(6×32=192, 256) = 4×256 = 1024 → floor(65536/1024) = 64 blocks
 *   The new result is more conservative and matches ncu / occupancy calculator.
 *
 * @param smMaxRegisters  Total 32-bit register file per SM
 * @param registersPerThread  Registers per thread declared by the kernel
 * @param threadsPerBlock  Block size in threads
 * @param warpSize  Threads per warp (always 32 on NVIDIA)
 * @param regAllocUnitPerWarp  Per-warp register allocation granularity (32-bit regs)
 * @returns Maximum concurrent blocks limited by register file
 */
function blocksByRegisters(
  smMaxRegisters: number,
  registersPerThread: number,
  threadsPerBlock: number,
  warpSize: number,
  regAllocUnitPerWarp: number
): number {
  const warpsPerBlock = Math.ceil(threadsPerBlock / warpSize);
  // Round each warp's register demand up to the allocation granularity.
  // This is the hardware allocation atom: the driver reserves registers in
  // multiples of regAllocUnitPerWarp (256 on Volta/Turing/Ampere/Ada/Hopper).
  const regsPerWarp = roundUp(registersPerThread * warpSize, regAllocUnitPerWarp);
  const regsPerBlock = warpsPerBlock * regsPerWarp;
  if (regsPerBlock <= 0) {
    return 0;
  }
  return Math.floor(smMaxRegisters / regsPerBlock);
}

/**
 * Computes occupancy breakdown for a given kernel configuration and GPU.
 *
 * Algorithm:
 * 1. For each limiting factor, calculate max concurrent blocks per SM
 * 2. Find the minimum (bottleneck) block count
 * 3. Calculate occupancy = (min_blocks * threads_per_block) / max_threads_per_sm
 * 4. Clamp occupancy to [0.0, 1.0] range
 *
 * The limiting_factor identifies which resource should be optimized first:
 * - "registers" is most common (high per-thread usage)
 * - "threads" suggests block size is too large
 * - "shared_mem" suggests memory allocation is too aggressive
 *
 * @param threadsPerBlock Threads per block (must be > 0)
 * @param sharedMemPerBlock Shared memory requested (bytes, can be 0)
 * @param registersPerThread Registers per thread (must be > 0)
 * @param spec GPU architecture limits (defaults to Ampere)
 * @returns Occupancy breakdown with limiting factor
 * @throws Error if threads_per_block or registers_per_thread <= 0
 */
export function computeOccupancyBreakdown(
  threadsPerBlock: number,
  sharedMemPerBlock: number,
  registersPerThread: number,
  spec: GpuSpec = AMPERE_LIKE_DEFAULT
): OccupancyBreakdown {
  // Validate inputs
  if (threadsPerBlock <= 0) {
    throw new Error("threads_per_block must be positive");
  }
  if (registersPerThread <= 0) {
    throw new Error("registers_per_thread must be positive");
  }

  const g = spec;
  // Use explicit smMaxWarps if available, otherwise compute from threads
  const maxWarps = g.smMaxWarps ?? Math.floor(g.smMaxThreads / g.warpSize);

  // Thread limit: blocks = max_threads / blocks_threads
  const blocksByThreads =
    threadsPerBlock > g.smMaxThreads
      ? 0
      : Math.floor(g.smMaxThreads / threadsPerBlock);

  // Warp limit: blocks = max_warps / warps_per_block
  const warpsPerBlock = Math.ceil(threadsPerBlock / g.warpSize);
  const blocksByWarps =
    warpsPerBlock > 0 ? Math.floor(maxWarps / warpsPerBlock) : 0;

  // Shared memory limit
  const blocksBySharedMem = Math.floor(
    blocksByShared(
      g.smMaxSharedMem,
      sharedMemPerBlock,
      g.smMaxBlocks,
      g.sharedMemAllocUnit,
      g.minSharedPerBlockAlloc
    )
  );

  // Register limit — per-warp granularity (Gap 7 fix; see blocksByRegisters)
  const blocksByRegs = Math.floor(
    blocksByRegisters(
      g.smMaxRegisters,
      registersPerThread,
      threadsPerBlock,
      g.warpSize,
      g.regAllocUnitPerWarp
    )
  );

  // Hard architectural block limit
  const blocksByBlockLimit = g.smMaxBlocks;

  // Find limiting factor (minimum block count)
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

  // Compute occupancy using active warps / max warps (NVIDIA occupancy convention)
  const activeWarpsPerSm = Math.min(blocksPerSm * warpsPerBlock, maxWarps);
  let occ = maxWarps > 0 ? activeWarpsPerSm / maxWarps : 0;
  occ = Math.max(0, Math.min(1, occ));  // Clamp to [0.0, 1.0]

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


/**
 * Collects occupancy-related warnings for a kernel configuration.
 *
 * Identifies potential issues with thread, register, and memory usage that could
 * limit performance through low occupancy or thread underutilization.
 *
 * Warnings triggered:
 * - threads < 128: Insufficient for effective latency hiding (each thread stalls ~400 cycles)
 * - threads < 64: Too low; queries occupancy logic
 * - threads % warp_size != 0: Wasted lanes (partial warp not fully utilized)
 * - warps per block < 4: Insufficient concurrency per block
 * - registers limiting: Register file is bottleneck (most common issue)
 * - registers > 64: Excessive per-thread usage
 * - shared_mem limiting: Shared memory is bottleneck
 * - occupancy < 0.4: Low occupancy means poor latency hiding capability
 *
 * @param threadsPerBlock Block size in threads
 * @param sharedMemPerBlock Shared memory allocation in bytes
 * @param registersPerThread Registers per thread
 * @param breakdown Previous occupancy calculation result
 * @param spec GPU architecture limits
 * @returns Array of unique warning messages
 */
export function collectWarnings(
  threadsPerBlock: number,
  sharedMemPerBlock: number,
  registersPerThread: number,
  breakdown: OccupancyBreakdown,
  spec: GpuSpec = AMPERE_LIKE_DEFAULT
): string[] {
  const ws = spec.warpSize;
  const warnings: string[] = [];

  // Gap 8: Kernel exceeds all SM limits — it cannot be launched at all.
  // All other warnings are moot; return immediately with a clear error.
  if (breakdown.blocks_per_sm === 0) {
    warnings.push(
      "Kernel cannot launch: combined register/shared-memory demand exceeds SM capacity — reduce registers or shared memory"
    );
    return warnings;
  }

  // Thread count heuristics
  if (threadsPerBlock < 128) {
    // 128 threads = 4 warps, considered minimum for effective latency hiding
    warnings.push("Low threads per block -> poor latency hiding");
  }
  if (threadsPerBlock < 64) {
    // 64 threads = 2 warps, almost never used in practice
    warnings.push("Very low threads per block (< 64) -> suspicious for occupancy");
  }
  if (threadsPerBlock % ws !== 0) {
    // Non-multiple of warp size leaves threads idle
    warnings.push("Threads not a multiple of warp size -> wasted lanes");
  }

  // Warp-level concurrency
  const activeWarps = threadsPerBlock / ws;
  if (activeWarps < 4) {
    // Recommend 4+ warps per block for reasonable SM utilization
    warnings.push("Too few warps per block -> poor SM utilization");
  }

  // Gap 10: Actionable register-pressure fix.
  // Compute the maximum regs/thread that would allow 2 blocks/SM, so the
  // developer has a concrete target for __launch_bounds__ rather than just
  // "check for over-allocation".
  if (breakdown.limiting_factor === "registers") {
    const warpsPerBlock = Math.ceil(threadsPerBlock / ws);
    const unit = spec.regAllocUnitPerWarp; // 256 on all modern NVIDIA GPUs
    // Budget per warp to fit 2 blocks: floor(smMaxRegisters / 2 / warpsPerBlock)
    const budgetPerWarp = Math.floor(
      Math.floor(spec.smMaxRegisters / 2) / warpsPerBlock
    );
    // Max regs/thread = floor(budgetPerWarp / warpSize), rounded down to
    // the per-thread granule (unit / warpSize = 8 on Ampere)
    const granule = Math.max(1, Math.floor(unit / ws));
    const targetRegs = Math.max(
      granule,
      Math.floor(Math.floor(budgetPerWarp / ws) / granule) * granule
    );
    warnings.push(
      `Register pressure limits occupancy — target ≤${targetRegs} regs/thread` +
        ` (add __launch_bounds__(${threadsPerBlock})) to fit 2 blocks/SM`
    );
  }
  if (registersPerThread > 64) {
    // >64 regs per thread is very high; suggests optimization opportunity
    warnings.push("Very high register usage per thread");
  }

  // Gap 10: Actionable shared-memory fix.
  if (breakdown.limiting_factor === "shared_mem") {
    const unit = spec.sharedMemAllocUnit;
    const targetShared =
      Math.floor(Math.floor(spec.smMaxSharedMem / 2) / unit) * unit;
    // Gap 11: also probe the binary-search margin so the message can quote a
    // specific shed amount when one would actually unlock a higher tier.
    const margin = findSharedMemPressureMargin(
      threadsPerBlock,
      sharedMemPerBlock,
      registersPerThread,
      spec
    );
    if (margin !== undefined) {
      warnings.push(
        `Shared memory limits occupancy — shed ≥${margin.margin} bytes/block ` +
          `(target ≤${sharedMemPerBlock - margin.margin} bytes) to reach ` +
          `'${margin.nextClass}' occupancy`
      );
    } else {
      warnings.push(
        `Shared memory limits occupancy — reduce to ≤${targetShared} bytes/block to fit 2 blocks/SM`
      );
    }
  }

  // Occupancy threshold
  if (breakdown.occupancy < 0.4) {
    // <40% occupancy: insufficient threads to hide memory latency
    warnings.push("Low occupancy -> likely poor latency hiding");
  }

  // Remove duplicates while preserving order
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
  /** Gap 9: "low" | "medium" | "high" classification of occupancy. */
  occupancy_class: string;
  /**
   * Minimum regs/thread reduction needed to reach the next occupancy tier.
   * Only populated when the limiting factor is "registers" and the current
   * tier is not already "high".  Computed by `findRegisterPressureMargin()`.
   * Example: 8 means reducing from 64 to 56 regs/thread would improve tier.
   */
  register_pressure_margin: number | undefined;
  /**
   * The occupancy tier reached after shedding `register_pressure_margin` regs/thread.
   * Values: "medium" | "high".  Undefined when `register_pressure_margin` is undefined.
   */
  next_occupancy_class: string | undefined;
  /**
   * B7: When the limiting factor switches after the register reduction (e.g. to
   * "shared_mem"), this field names the new bottleneck.  The register-reduction
   * advice is still valid — the tier improvement is real — but the user should
   * also address this resource to improve occupancy further.
   * Undefined when the limiting factor stays "registers" after reduction, or
   * when `register_pressure_margin` is undefined.
   */
  next_limiting_factor: LimitName | undefined;
  /**
   * Gap 11: Minimum shared-memory reduction (bytes/block) needed to reach the
   * next occupancy tier.  Only populated when the limiting factor is
   * "shared_mem" and the current tier is not already "high".  Computed by
   * `findSharedMemPressureMargin()`.
   * Example: 4096 means reducing per-block shared memory by 4 KB would
   * improve the occupancy tier.
   */
  shared_mem_pressure_margin: number | undefined;
  /**
   * Gap 11: The occupancy tier reached after shedding
   * `shared_mem_pressure_margin` bytes/block.  Values: "medium" | "high".
   * Undefined when `shared_mem_pressure_margin` is undefined.
   */
  next_occupancy_class_shared: string | undefined;
  /**
   * Gap 11: When the limiting factor switches after the shared-memory
   * reduction (e.g. to "registers"), this field names the new bottleneck.
   * The shared-memory advice is still valid — the tier improvement is real —
   * but the user should also address this resource for further gains.
   */
  next_limiting_factor_shared: LimitName | undefined;
  limiting_factor: LimitName;
  blocks_per_sm: number;
  limits: Record<string, number>;
  warp_metrics: Record<string, number>;
  waste_metrics: Record<string, number>;
  warnings: string[];
  /**
   * Gap 11: Human-readable device-level throughput estimate.
   * "N / M SMs active" when smCount is known; undefined otherwise.
   * Example: "54 / 108 SMs active" means 50% of the A100's SMs are busy.
   * I3: when `analyzeKernel` receives `gridBlocks`, N is capped by that launch
   * size so tiny grids cannot exceed concurrent block count.
   */
  estimated_sm_utilization: string | undefined;
}

/** Optional knobs for `analyzeKernel` (device / launch context). */
export interface AnalyzeKernelOptions {
  /** Total thread blocks in the launch; caps reported SM participation (I3). */
  gridBlocks?: number;
}

function occupancyClassRank(cls: string): number {
  if (cls === "low") {
    return 0;
  }
  if (cls === "medium") {
    return 1;
  }
  return 2;
}

/**
 * Finds the minimum register reduction that moves a kernel to a higher occupancy tier.
 *
 * When a kernel is register-limited, this function uses a binary search over
 * `registersPerThread` values to find the lowest register count at which
 * `classifyOccupancy()` returns a strictly higher tier ("low" → "medium", or
 * "medium" → "high").
 *
 * Algorithm:
 * 1. Bail out early if the kernel is not register-limited, is already at "high",
 *    or has only 1 register (cannot reduce further).
 * 2. Binary search in [1, registersPerThread - 1] for the largest `mid` value
 *    whose occupancy tier exceeds the current tier.  This maximises `margin`
 *    (i.e., we want the minimum shed amount, so we want the highest `mid` that
 *    still improves the tier).
 * 3. Return `{ margin: registersPerThread - best, nextClass }` so the caller can
 *    display a concrete target (e.g., "shed 8 regs/thread to reach 'high'").
 *
 * Returns `undefined` when no reduction within the [1, rpt-1] range improves
 * the occupancy tier (e.g., shared memory is the real bottleneck).
 *
 * @param threadsPerBlock  Block size in threads
 * @param sharedMemPerBlock  Per-block shared memory in bytes
 * @param registersPerThread  Current registers/thread declared by the kernel
 * @param spec  GPU architecture limits (defaults to Ampere)
 * @returns `{ margin, nextClass }` or `undefined` when no improvement is reachable
 */
export function findRegisterPressureMargin(
  threadsPerBlock: number,
  sharedMemPerBlock: number,
  registersPerThread: number,
  spec: GpuSpec = AMPERE_LIKE_DEFAULT
): { margin: number; nextClass: string; limitingSwitchesTo: LimitName | undefined } | undefined {
  const current = computeOccupancyBreakdown(
    threadsPerBlock,
    sharedMemPerBlock,
    registersPerThread,
    spec
  );
  const currentClass = classifyOccupancy(current.occupancy);
  const currentRank = occupancyClassRank(currentClass);

  if (current.limiting_factor !== "registers" || currentRank >= 2 || registersPerThread <= 1) {
    return undefined;
  }

  // I2: search in real allocation granularity steps, not 1-reg increments.
  // Registers are allocated per warp in `regAllocUnitPerWarp` chunks, so the
  // effective per-thread step is `regAllocUnitPerWarp / warpSize` (8 regs on
  // modern architectures with 256/32). Returning a non-granular margin would
  // be optimistic by up to one step.
  const granule = Math.max(1, Math.floor(spec.regAllocUnitPerWarp / spec.warpSize));
  const effectiveCurrent =
    Math.floor(registersPerThread / granule) * granule;
  const maxCandidate = Math.floor((registersPerThread - 1) / granule) * granule;
  let lo = granule;
  let hi = maxCandidate;
  let best: number | undefined;
  let bestClass: string | undefined;

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / (2 * granule)) * granule;
    const trial = computeOccupancyBreakdown(
      threadsPerBlock,
      sharedMemPerBlock,
      mid,
      spec
    );
    const trialClass = classifyOccupancy(trial.occupancy);
    const improved = occupancyClassRank(trialClass) > currentRank;
    if (improved) {
      best = mid;
      bestClass = trialClass;
      lo = mid + granule;
    } else {
      hi = mid - granule;
    }
  }

  if (best === undefined || bestClass === undefined) {
    return undefined;
  }

  // B7: verify the trial breakdown at `best` to detect a limiting-factor switch.
  // The tier improvement is genuine and the margin advice is still valid — the
  // user CAN reduce registers by `margin` to reach `nextClass`.  However, if the
  // new limiting factor is no longer "registers" (e.g. shared_mem takes over),
  // the caller should surface this so the user knows the NEXT bottleneck to
  // address after the register reduction.
  const trialAtBest = computeOccupancyBreakdown(
    threadsPerBlock,
    sharedMemPerBlock,
    best,
    spec
  );
  const limitingSwitchesTo: LimitName | undefined =
    trialAtBest.limiting_factor !== "registers"
      ? trialAtBest.limiting_factor
      : undefined;

  const margin = Math.max(0, effectiveCurrent - best);
  if (margin === 0) {
    return undefined;
  }
  return {
    margin,
    nextClass: bestClass,
    limitingSwitchesTo,
  };
}

/**
 * Gap 11: Finds the minimum shared-memory reduction that moves a kernel to a
 * higher occupancy tier — the symmetric counterpart to
 * `findRegisterPressureMargin` for shared-memory-limited kernels.
 *
 * Algorithm mirrors the register-pressure margin search:
 * 1. Bail out early when the kernel is not shared-mem-limited, is already
 *    at "high", or has no shared memory to shed.
 * 2. Binary search in [0, sharedMemPerBlock - allocUnit] for the largest
 *    `mid` byte allocation whose occupancy tier exceeds the current tier.
 *    `mid` is rounded down to the allocation granule (typically 256 bytes)
 *    so the recommendation matches what the driver will actually allocate.
 * 3. Return `{ margin: bytes-to-shed, nextClass, limitingSwitchesTo? }`.
 *
 * Returns `undefined` when no reduction within the [0, current-allocUnit]
 * range improves the tier (e.g. registers are the real bottleneck).
 *
 * @param threadsPerBlock  Block size in threads
 * @param sharedMemPerBlock  Per-block shared memory in bytes
 * @param registersPerThread  Registers per thread
 * @param spec  GPU architecture limits (defaults to Ampere)
 * @returns `{ margin, nextClass, limitingSwitchesTo }` or undefined
 */
export function findSharedMemPressureMargin(
  threadsPerBlock: number,
  sharedMemPerBlock: number,
  registersPerThread: number,
  spec: GpuSpec = AMPERE_LIKE_DEFAULT
): { margin: number; nextClass: string; limitingSwitchesTo: LimitName | undefined } | undefined {
  const current = computeOccupancyBreakdown(
    threadsPerBlock,
    sharedMemPerBlock,
    registersPerThread,
    spec
  );
  const currentClass = classifyOccupancy(current.occupancy);
  const currentRank = occupancyClassRank(currentClass);

  if (
    current.limiting_factor !== "shared_mem" ||
    currentRank >= 2 ||
    sharedMemPerBlock <= 0
  ) {
    return undefined;
  }

  const allocUnit = Math.max(1, spec.sharedMemAllocUnit);
  // Search space: round `sharedMemPerBlock` down to the allocation granule
  // and search [0, capStep - 1] (in granule units).  The trial value is
  // `step * allocUnit`, ensuring all probes are valid driver allocations.
  const capStep = Math.floor(sharedMemPerBlock / allocUnit);
  if (capStep <= 0) {
    return undefined;
  }

  let lo = 0;
  let hi = capStep - 1;
  let best: number | undefined;
  let bestClass: string | undefined;

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    const trialBytes = mid * allocUnit;
    const trial = computeOccupancyBreakdown(
      threadsPerBlock,
      trialBytes,
      registersPerThread,
      spec
    );
    const trialClass = classifyOccupancy(trial.occupancy);
    const improved = occupancyClassRank(trialClass) > currentRank;
    if (improved) {
      best = trialBytes;
      bestClass = trialClass;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  if (best === undefined || bestClass === undefined) {
    return undefined;
  }

  const trialAtBest = computeOccupancyBreakdown(
    threadsPerBlock,
    best,
    registersPerThread,
    spec
  );
  const limitingSwitchesTo: LimitName | undefined =
    trialAtBest.limiting_factor !== "shared_mem"
      ? trialAtBest.limiting_factor
      : undefined;

  return {
    margin: sharedMemPerBlock - best,
    nextClass: bestClass,
    limitingSwitchesTo,
  };
}

export function analyzeKernel(
  threadsPerBlock: number,
  sharedMemPerBlock: number,
  registersPerThread: number,
  spec: GpuSpec = AMPERE_LIKE_DEFAULT,
  opts?: AnalyzeKernelOptions
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
  const threadsPerSm = Math.min(
    bd.blocks_per_sm * threadsPerBlock,
    g.smMaxThreads
  );
  const threadOccupancy =
    g.smMaxThreads > 0 ? threadsPerSm / g.smMaxThreads : 0;
  const warpOccupancy =
    maxWarpsPerSm > 0 ? warpsPerSm / maxWarpsPerSm : 0;

  const regsPerBlockRaw = registersPerThread * threadsPerBlock;
  // Gap 7 fix: use per-warp allocation granularity to match blocksByRegisters()
  const warpsPerBlockForWaste = Math.ceil(threadsPerBlock / g.warpSize);
  const regsPerWarpAlloc = roundUp(registersPerThread * g.warpSize, g.regAllocUnitPerWarp);
  const regsPerBlockAlloc = warpsPerBlockForWaste * regsPerWarpAlloc;
  const sharedPerBlockAlloc =
    sharedMemPerBlock <= 0
      ? 0
      : Math.max(
          roundUp(sharedMemPerBlock, g.sharedMemAllocUnit),
          g.minSharedPerBlockAlloc
        );

  const regsUsedPerSm = Math.min(
    bd.blocks_per_sm * regsPerBlockAlloc,
    g.smMaxRegisters
  );
  const sharedUsedPerSm = Math.min(
    bd.blocks_per_sm * sharedPerBlockAlloc,
    g.smMaxSharedMem
  );
  const registerMargin = findRegisterPressureMargin(
    threadsPerBlock,
    sharedMemPerBlock,
    registersPerThread,
    spec
  );
  // Gap 11: parallel margin for shared-memory-limited kernels.
  const sharedMemMargin = findSharedMemPressureMargin(
    threadsPerBlock,
    sharedMemPerBlock,
    registersPerThread,
    spec
  );

  return {
    gpu: g.name,
    threads_per_block: threadsPerBlock,
    shared_mem_per_block: sharedMemPerBlock,
    registers_per_thread: registersPerThread,
    occupancy_threads: Math.round(threadOccupancy * 1e6) / 1e6,
    occupancy_warps: Math.round(warpOccupancy * 1e6) / 1e6,
    occupancy: Math.round(warpOccupancy * 1e6) / 1e6,
    // Gap 9: expose occupancy tier so callers don’t have to re-implement the
    // low/medium/high bucketing themselves.
    occupancy_class: classifyOccupancy(warpOccupancy),
    register_pressure_margin: registerMargin?.margin,
    next_occupancy_class: registerMargin?.nextClass,
    next_limiting_factor: registerMargin?.limitingSwitchesTo,
    shared_mem_pressure_margin: sharedMemMargin?.margin,
    next_occupancy_class_shared: sharedMemMargin?.nextClass,
    next_limiting_factor_shared: sharedMemMargin?.limitingSwitchesTo,
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
    waste_metrics: {
      unused_threads_per_sm: Math.max(0, g.smMaxThreads - threadsPerSm),
      unused_warps_per_sm: Math.max(0, maxWarpsPerSm - warpsPerSm),
      unused_registers_per_sm: Math.max(0, g.smMaxRegisters - regsUsedPerSm),
      unused_shared_mem_bytes_per_sm: Math.max(
        0,
        g.smMaxSharedMem - sharedUsedPerSm
      ),
      allocated_registers_per_block: regsPerBlockAlloc,
      allocated_shared_mem_bytes_per_block: sharedPerBlockAlloc,
    },
    warnings: warns,
    // Gap 11: surface the device-level picture.  Per-SM occupancy alone hides
    // whether the kernel will saturate a 4-SM laptop GPU or a 108-SM A100.
    // When smCount is not known (SASS-only without nvidia-smi), return undefined.
    estimated_sm_utilization:
      g.smCount !== undefined
        ? (() => {
            const idealSms = Math.round(warpOccupancy * g.smCount);
            const rawGrid = opts?.gridBlocks;
            const gridCap =
              rawGrid !== undefined &&
              Number.isFinite(rawGrid) &&
              rawGrid >= 0
                ? Math.floor(rawGrid)
                : undefined;
            const displayed =
              gridCap !== undefined ? Math.min(idealSms, gridCap) : idealSms;
            const gridNote =
              gridCap !== undefined ? `; grid≤${gridCap} blocks` : "";
            return `${displayed} / ${g.smCount} SMs active${gridNote}`;
          })()
        : undefined,
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
