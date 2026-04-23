/**
 * GPU architecture specifications and configuration management.
 *
 * This module provides:
 * 1. **Architecture Table** (GPU_SM_CONFIGS): Per-SM resource limits for each
 *    NVIDIA compute capability (CC 7.0-12.0, covering Volta through Blackwell)
 * 2. **SM Count Detection**: Queries nvidia-smi for GPU presence and device count
 * 3. **Preset Resolution**: Converts flexible config strings (e.g., "auto", "a100")
 *    to complete GpuSpec with both architecture and device count
 * 4. **Occupancy Analysis**: Enables occupancy calculations with authoritative limits
 *
 * Architecture dimensions defined:
 * - smMaxThreads: Max concurrent threads per SM (GPU-dependent: 1024-2048)
 * - smMaxBlocks: Max concurrent blocks per SM (16-32)
 * - smMaxRegisters: Register file size per SM (64K 32-bit regs)
 * - smMaxSharedMem: Shared memory per SM (64 KB - 228 KB)
 * - warpSize: Threads per warp (always 32 for modern NVIDIA)
 * - Allocation units: Granularity of shared memory and register allocation
 *
 * Example use:
 * - resolveGpuSpecForAnalysis("auto") -> probes device and adapts
 * - gpuSpecFromArch("8.0", 108) -> A100 configuration (CC 8.0, 108 SMs)
 * - getPreset("rtx-4090") -> RTX 4090 preset (CC 8.9, 128 SMs)
 */

import {
  queryFirstGpuFromNvidiaSmi,
  RESOLUTION_SMI_NOT_PRESENT,
  RESOLUTION_SMI_PROBE_FAILED,
} from "./nvidia_smi";
import { detectSmCountLocally } from "./local_cuda_detect";

/**
 * Complete GPU specification: architecture limits + device info.
 *
 * Architecture limits (per SM) determine occupancy and resource constraints.
 * smCount is device-specific (number of SMs on the GPU).
 * Together they enable occupancy calculation for a specific hardware/kernel pair.
 */
export interface GpuSpec {
  name: string;              // Human-readable GPU/arch name (e.g., "ampere-sm80")
  smMaxThreads: number;      // Max threads per SM (threads/block constraint)
  smMaxBlocks: number;       // Max concurrent blocks per SM (hard limit)
  smMaxSharedMem: number;    // Total shared memory per SM (bytes)
  smMaxRegisters: number;    // Total register file per SM (32-bit registers)
  warpSize: number;          // Threads per warp (32 on modern NVIDIA)
  smMaxWarps: number | undefined;  // Max warps per SM (explicit or computed)
  regAllocUnitPerBlock: number;   // Register allocation granule per block (32-bit regs)
  regAllocUnitPerWarp: number;    // Register allocation granule per warp (bytes)
  sharedMemAllocUnit: number;     // Shared memory allocation granule (bytes)
  minSharedPerBlockAlloc: number; // Minimum shared allocation per block (bytes)
  smCount: number | undefined;    // Number of SMs on device (device-specific)
}

/**
 * Architecture limits without device SM count.
 * Used for architecture table entries (smCount added per-device).
 */
export type GpuArchLimits = Omit<GpuSpec, "smCount">;

/**
 * Supported compute capability versions (keys for architecture table).
 * Maps to NVIDIA architecture generations:
 * 7.0: Volta (V100, etc.)
 * 7.5: Turing (T4, RTX 20 series, etc.)
 * 8.0: Ampere (A100, etc.)
 * 8.6: Ampere mobile/small (RTX 30 desktop series limit, etc.)
 * 8.9: Ada (RTX 40 series)
 * 9.0: Hopper (H100, H200)
 * 10.0: Blackwell datacenter (B200, GB200)
 * 12.0: Blackwell consumer (RTX 50 series)
 */
export type GpuComputeCapabilityKey =
  | "7.0"
  | "7.5"
  | "8.0"
  | "8.6"
  | "8.9"
  | "9.0"
  | "10.0"
  | "12.0";

/**
 * Architecture specification table: per-SM resource limits for each compute capability.
 *
 * Values from:
 * - NVIDIA CUDA Programming Guide (compute capability appendix)
 * - NVIDIA Tuning Guides (architecture-specific)
 *
 * Key design notes:
 * - All modern GPUs: warpSize = 32, regAllocUnitPerBlock = 256, sharedMemAllocUnit = 256
 * - Maxwell+ (cc 5.3): Shared memory allocation minimum (minSharedPerBlockAlloc)
 * - Ampere+ (cc 8.0): Support for dynamic shared memory
 *
 * Per compute capability (CC):
 * - Volta (CC 7.0): 2048 threads/SM, 64 W/SM, 96 KB shared, 65K regs (HPC-focused)
 * - Turing (CC 7.5): 1024 threads/SM, 32 W/SM, 64 KB shared (cost/efficiency trade-off)
 * - Ampere (CC 8.0): 2048 threads/SM, 64 W/SM, 164 KB shared (high-end datacenter)
 * - Ampere Mobile (CC 8.6): 1536 threads/SM, 48 W/SM, 100 KB shared (mobile/RTX 30)
 * - Ada (CC 8.9): 1536 threads/SM, 48 W/SM, 100 KB shared (efficiency-optimized)
 * - Hopper (CC 9.0): 2048 threads/SM, 64 W/SM, 228 KB shared (Transformer scaling)
 * - Blackwell DC (CC 10.0): 2048 threads/SM, 64 W/SM, 228 KB shared (Hopper-equivalent)
 * - Blackwell Consumer (CC 12.0): 1536 threads/SM, 48 W/SM, 100 KB shared (RTX 50 series)
 */
export const GPU_SM_CONFIGS: Record<GpuComputeCapabilityKey, GpuArchLimits> = {
  "7.0": {
    name: "volta-sm70",
    smMaxThreads: 2048,
    smMaxWarps: 64,
    warpSize: 32,
    smMaxBlocks: 32,
    smMaxRegisters: 65536,
    regAllocUnitPerBlock: 256,
    regAllocUnitPerWarp: 256,
    smMaxSharedMem: 96 * 1024,
    sharedMemAllocUnit: 256,
    minSharedPerBlockAlloc: 256,
  },
  "7.5": {
    name: "turing-sm75",
    smMaxThreads: 1024,
    smMaxWarps: 32,
    warpSize: 32,
    smMaxBlocks: 16,
    smMaxRegisters: 65536,
    regAllocUnitPerBlock: 256,
    regAllocUnitPerWarp: 256,
    smMaxSharedMem: 64 * 1024,
    sharedMemAllocUnit: 256,
    minSharedPerBlockAlloc: 256,
  },
  "8.0": {
    name: "ampere-sm80",
    smMaxThreads: 2048,
    smMaxWarps: 64,
    warpSize: 32,
    smMaxBlocks: 32,
    smMaxRegisters: 65536,
    regAllocUnitPerBlock: 256,
    regAllocUnitPerWarp: 256,
    smMaxSharedMem: 164 * 1024,
    sharedMemAllocUnit: 256,
    minSharedPerBlockAlloc: 256,
  },
  "8.6": {
    name: "ampere-sm86",
    smMaxThreads: 1536,
    smMaxWarps: 48,
    warpSize: 32,
    smMaxBlocks: 16,
    smMaxRegisters: 65536,
    regAllocUnitPerBlock: 256,
    regAllocUnitPerWarp: 256,
    smMaxSharedMem: 100 * 1024,
    sharedMemAllocUnit: 256,
    minSharedPerBlockAlloc: 256,
  },
  "8.9": {
    name: "ada-sm89",
    smMaxThreads: 1536,
    smMaxWarps: 48,
    warpSize: 32,
    smMaxBlocks: 24,
    smMaxRegisters: 65536,
    regAllocUnitPerBlock: 256,
    regAllocUnitPerWarp: 256,
    smMaxSharedMem: 100 * 1024,
    sharedMemAllocUnit: 256,
    minSharedPerBlockAlloc: 256,
  },
  "9.0": {
    name: "hopper-sm90",
    smMaxThreads: 2048,
    smMaxWarps: 64,
    warpSize: 32,
    smMaxBlocks: 32,
    smMaxRegisters: 65536,
    regAllocUnitPerBlock: 256,
    regAllocUnitPerWarp: 256,
    smMaxSharedMem: 228 * 1024,
    sharedMemAllocUnit: 256,
    minSharedPerBlockAlloc: 256,
  },
  /**
   * Blackwell CC 10.0 (e.g. B200/GB200, `.target sm_100`, SMI compute_cap 10.0).
   * Occupancy-related limits from NVIDIA Blackwell Tuning Guide §1.4.1.1 (concurrent warps 64,
   * blocks/SM 32, 228 KB shared/SM, 64K 32-bit regs/SM). Max threads/SM = 64×32 = 2048.
   * regAllocUnitPerWarp / sharedMemAllocUnit / minSharedPerBlockAlloc follow Hopper-style 256 B
   * (verify via CUDA Programming Guide — compute capabilities if you need exact granularity).
   */
  "10.0": {
    name: "blackwell-sm100",
    smMaxThreads: 2048,
    smMaxWarps: 64,
    warpSize: 32,
    smMaxBlocks: 32,
    smMaxRegisters: 65536,
    regAllocUnitPerBlock: 256,
    regAllocUnitPerWarp: 256,
    smMaxSharedMem: 228 * 1024,
    sharedMemAllocUnit: 256,
    minSharedPerBlockAlloc: 256,
  },
  /**
   * Blackwell consumer CC 12.0 (RTX 50 series / RTX PRO 4000 Blackwell, `.target sm_120`,
   * SMI compute_cap 12.0). Occupancy limits match GB20x architecture: 48 warps/SM, 1536
   * threads/SM, 24 blocks/SM, 65536 32-bit regs/SM, 100 KB shared/SM.
   * (verify via CUDA Programming Guide — compute capabilities if you need exact granularity).
   */
  "12.0": {
    name: "blackwell-sm120",
    smMaxThreads: 1536,
    smMaxWarps: 48,
    warpSize: 32,
    smMaxBlocks: 24,
    smMaxRegisters: 65536,
    regAllocUnitPerBlock: 256,
    regAllocUnitPerWarp: 256,
    smMaxSharedMem: 100 * 1024,
    sharedMemAllocUnit: 256,
    minSharedPerBlockAlloc: 256,
  },
};

export const GPU_COMPUTE_CAPABILITY_KEYS = Object.keys(
  GPU_SM_CONFIGS
) as GpuComputeCapabilityKey[];

/** Default fallback compute capability when PTX and SMI targets are unavailable. */
export const DEFAULT_FALLBACK_CC: GpuComputeCapabilityKey = "8.6";

export const AMPERE_LIKE_DEFAULT: GpuSpec = {
  ...GPU_SM_CONFIGS[DEFAULT_FALLBACK_CC],
  smCount: undefined,
};

export interface GpuPresetMetadata {
  key: string;
  label: string;
  cc: GpuComputeCapabilityKey;
  smTag: string;
  smCount: number | undefined;
}

/** Named preset configurations (human-friendly config values). */
const PRESET_METADATA: Record<string, Omit<GpuPresetMetadata, "key">> = {
  "ampere-like-default": {
    label: "Ampere-like default (SM86)",
    cc: "8.6",
    smTag: "sm_86",
    smCount: undefined,
  },
  a100: {
    label: "A100 (SM80)",
    cc: "8.0",
    smTag: "sm_80",
    smCount: 108,
  },
  "rtx-4090": {
    label: "RTX 4090 (SM89)",
    cc: "8.9",
    smTag: "sm_89",
    smCount: 128,
  },
  "h100-sxm": {
    label: "H100 SXM (SM90)",
    cc: "9.0",
    smTag: "sm_90",
    smCount: 132,
  },
  "h100-pcie": {
    label: "H100 PCIe (SM90)",
    cc: "9.0",
    smTag: "sm_90",
    smCount: 114,
  },
  h200: {
    label: "H200 (SM90)",
    cc: "9.0",
    smTag: "sm_90",
    smCount: 132,
  },
  b200: {
    label: "B200 (SM100)",
    cc: "10.0",
    smTag: "sm_100",
    smCount: 192,
  },
  gb200: {
    label: "GB200 (SM100)",
    cc: "10.0",
    smTag: "sm_100",
    smCount: 192,
  },
  "blackwell-consumer-default": {
    label: "Blackwell consumer default (SM120)",
    cc: "12.0",
    smTag: "sm_120",
    smCount: undefined,
  },
};

const PRESET_ARCH: Record<string, GpuComputeCapabilityKey> = Object.fromEntries(
  Object.entries(PRESET_METADATA).map(([key, meta]) => [key, meta.cc])
) as Record<string, GpuComputeCapabilityKey>;

/** Named preset SM counts (device-specific). */
const PRESET_SM_COUNT: Record<string, number | undefined> = Object.fromEntries(
  Object.entries(PRESET_METADATA).map(([key, meta]) => [key, meta.smCount])
) as Record<string, number | undefined>;

export const KNOWN_PRESET_KEYS = Object.keys(PRESET_METADATA).sort() as string[];

export function getPresetMetadata(name: string): GpuPresetMetadata {
  const key = name.trim().toLowerCase();
  const meta = PRESET_METADATA[key];
  if (!meta) {
    throw new Error(
      `Unknown GPU preset ${JSON.stringify(name)}; known: ${KNOWN_PRESET_KEYS.join(", ")}`
    );
  }
  return {
    key,
    ...meta,
  };
}

export function allPresetMetadata(): GpuPresetMetadata[] {
  return KNOWN_PRESET_KEYS.map((k) => ({ key: k, ...PRESET_METADATA[k]! }));
}

export function presetForSmVersion(sm: number): GpuPresetMetadata | undefined {
  const smTag = `sm_${sm}`;
  const matches = allPresetMetadata().filter((p) => p.smTag === smTag);
  if (matches.length === 0) {
    return undefined;
  }
  if (matches.length === 1) {
    return matches[0];
  }
  if (sm === 90) {
    return matches.find((p) => p.key === "h100-sxm") ?? matches[0];
  }
  if (sm === 100) {
    return matches.find((p) => p.key === "b200") ?? matches[0];
  }
  // Prefer architecture defaults when multiple product presets share the same SM.
  return (
    matches.find((p) =>
      p.key.includes("default") || p.key === "ampere-like-default"
    ) ?? matches[0]
  );
}

/**
 * GPU name pattern -> SM count heuristic.
 * Used when nvidia-smi reports device name but not SM count.
 * Applied in order; first match wins.
 */
interface NameToSmCountRule {
  pattern: RegExp;
  smCount: number;
}

/**
 * Heuristic rules to infer SM count from device name string.
 * Updated as new GPU models appear; covers datacenter (A-series) and consumer (RTX) lines.
 */
const NAME_TO_SM_COUNT_RULES: readonly NameToSmCountRule[] = [
  // Datacenter A-series
  { pattern: /\bA100\b/i, smCount: 108 },
  { pattern: /\bA30\b/i, smCount: 56 },
  { pattern: /\bA40\b/i, smCount: 84 },
  { pattern: /\bA10\b/i, smCount: 72 },
  
  // L-series (learning)
  { pattern: /\bL4\b/i, smCount: 58 },
  { pattern: /\bL40S\b/i, smCount: 142 },
  { pattern: /\bL40\b/i, smCount: 142 },
  
  // Older datacenter
  { pattern: /\bV100\b/i, smCount: 80 },
  { pattern: /\bT4\b/i, smCount: 40 },
  
  // Consumer RTX 40 series (Ampere generation)
  { pattern: /\bRTX\s*4090\b/i, smCount: 128 },
  { pattern: /\bRTX\s*4080\b/i, smCount: 76 },
  { pattern: /\bRTX\s*4070\s*Ti\b/i, smCount: 60 },
  { pattern: /\bRTX\s*4070\b/i, smCount: 46 },
  
  // Consumer RTX 30 series (Ampere generation)
  { pattern: /\bRTX\s*3090\b/i, smCount: 82 },
  { pattern: /\bRTX\s*3080\b/i, smCount: 68 },
  { pattern: /\bRTX\s*3070\b/i, smCount: 46 },
  { pattern: /\bRTX\s*3060\b/i, smCount: 28 },
  
  // H-series (Hopper datacenter)
  { pattern: /\bH100\b/i, smCount: 120 },
  { pattern: /\bH200\b/i, smCount: 132 },
  
  // B-series (Blackwell datacenter)
  { pattern: /\bB200\b/i, smCount: 192 },
];

/**
 * Infers SM count from GPU name using regex pattern matching.
 * Returns undefined if no rule matches.
 *
 * Example: "NVIDIA A100-PCIE-40GB" -> 108
 *
 * @param name GPU name string from nvidia-smi or CUDA API
 * @returns SM count if recognized, else undefined
 */
export function inferSmCountFromGpuName(name: string): number | undefined {
  const model = name.trim();
  if (!model) {
    return undefined;
  }
  const rule = NAME_TO_SM_COUNT_RULES.find((r) => r.pattern.test(model));
  return rule?.smCount;
}

/**
 * Internal: Extracts SM count from nvidia-smi query result.
 *
 * nvidia-smi can provide multiprocessor_count (number of SMs).
 * If available, that's our most authoritative source.
 * Also tracks which source provided the SM count (for resolution string).
 *
 * @param probed Result from queryFirstGpuFromNvidiaSmi()
 * @returns SM count + source attribution
 */
function resolveSmCountAndSource(
  probed: Awaited<ReturnType<typeof queryFirstGpuFromNvidiaSmi>>
): {
  smCount: number | undefined;
  source:
    | "smi.multiprocessor_count"
    | "name.heuristic"
    | "local.cache"
    | "local.cuda.detect"
    | "unknown";
} {
  if (!probed.ok) {
    return { smCount: undefined, source: "unknown" };
  }
  if (probed.row.multiprocessor_count != null) {
    return {
      smCount: probed.row.multiprocessor_count,
      source: "smi.multiprocessor_count",
    };
  }
  return { smCount: undefined, source: "unknown" };
}

/**
 * Loads a named GPU preset configuration.
 *
 * Known presets (case-insensitive):
 * - "ampere-like-default": CC 8.6, undefined SM count
 * - "a100": CC 8.0, 108 SMs
 * - "rtx-4090": CC 8.9, 128 SMs
 *
 * Throws error if preset not found.
 *
 * @param name Preset name (case-insensitive)
 * @returns Corresponding GpuSpec
 * @throws Error if name not in presets
 */
export function getPreset(name: string): GpuSpec {
  const meta = getPresetMetadata(name);
  return gpuSpecFromArch(meta.cc, meta.smCount);
}

/**
 * Constructs GpuSpec from architecture key and optional SM count.
 *
 * Copies all architecture limits from GPU_SM_CONFIGS[cc] and adds device SM count.
 * Used when you have compute capability and SM count independently.
 *
 * @param cc Compute capability key (7.0, 8.0, 8.6, etc.)
 * @param smCount Optional: number of SMs on device (undefined if unknown)
 * @returns Complete GpuSpec combining architecture + device info
 */
export function gpuSpecFromArch(
  cc: GpuComputeCapabilityKey,
  smCount: number | undefined
): GpuSpec {
  return { ...GPU_SM_CONFIGS[cc], smCount: smCount ?? undefined };
}

/**
 * Maps SM version numbers from NVIDIA to our compute capability keys.
 * SM version = last 2 or 3 digits from `.target sm_XX` directive.
 * Example: sm_80 -> 80 -> "8.0" (Ampere)
 */
const SM_VERSION_TO_CC: Partial<Record<number, GpuComputeCapabilityKey>> = {
  70: "7.0",  // Volta
  75: "7.5",  // Turing
  80: "8.0",  // Ampere
  86: "8.6",  // Ampere mobile
  87: "8.6",  // Ampere mobile variant (maps to 8.6)
  89: "8.9",  // Ada
  90: "9.0",  // Hopper
  /** CC 10.0 datacenter Blackwell (B200/GB200). */
  100: "10.0",
  /** CC 12.0 consumer Blackwell (RTX 50 series / RTX PRO 4000 Blackwell). */
  120: "12.0",
};

/**
 * Extracts compute capability from PTX `.target` directive and maps to architecture key.
 *
 * Process:
 * 1. Parse PTX text for `.target sm_XX` directive (regex match)
 * 2. Convert sm_XX number to compute capability key
 * 3. Return key or undefined if parsing fails
 *
 * Example: "... .target sm_80 ..." -> "8.0" (Ampere)
 *
 * @param ptx PTX text containing `.target` directive
 * @returns Compute capability key, or undefined if not found or unrecognized
 */
export function ccKeyFromPtxTarget(ptx: string): GpuComputeCapabilityKey | undefined {
  const sm = parsePtxSmTargetVersion(ptx);
  if (sm === undefined) {
    return undefined;
  }
  return SM_VERSION_TO_CC[sm];
}

/**
 * Converts nvidia-smi's `compute_cap` string (e.g., "8.0") to architecture key.
 *
 * nvidia-smi reports compute capability as major.minor (e.g., "8.0", "8.6").
 * This function parses and validates, returning corresponding architecture table key.
 *
 * Example: "8.0" -> "8.0" (Ampere high-end)
 * Example: "8.6" -> "8.6" (Ampere mobile)
 * Example: "9.9" -> undefined (unsupported version)
 *
 * @param s Compute capability string from nvidia-smi
 * @returns Architecture key if recognized, else undefined
 */
export function computeCapToArchKey(s: string): GpuComputeCapabilityKey | undefined {
  const n = parseFloat(s.trim());
  if (!Number.isFinite(n)) {
    return undefined;
  }
  const k = n.toFixed(1) as GpuComputeCapabilityKey;
  return k in GPU_SM_CONFIGS ? k : undefined;
}

/**
 * Result of GPU specification resolution with explanation string.
 * Used to provide users with details about how the config was determined.
 */
export interface GpuSpecResolution {
  spec: GpuSpec;          // Final GPU specification (architecture + SM count)
  resolution: string;     // Human-readable explanation of how spec was determined
}

/**
 * Options for GPU specification resolution.
 * Allows fallback strategies when primary sources are unavailable.
 */
export interface ResolveGpuSpecOptions {
  /** 
   * When set and contains `.target sm_XX`, uses PTX compute capability.
   * In "auto" mode, this drives architecture selection while SM count comes from nvidia-smi.
   */
  ptxText?: string;
  
  /** 
   * When true and nvidia-smi cannot provide SM count, attempt one-shot local CUDA 
   * compilation/run for direct SM count detection. Uses cache to avoid re-running.
   * Fallback strategy when nvidia-smi is unavailable.
   */
  enableLocalCudaDetect?: boolean;
}

/**
 * Resolves complete GPU specification from multiple sources in priority order.
 *
 * Preset Mode (presetKey != "auto"):
 * - Directly returns named preset (e.g., "a100", "rtx-4090")
 * - No device probing needed
 * - Resolution string: "preset:{key}"
 *
 * Auto Mode (presetKey == "auto"):
 * - Attempts multi-source detection:
 *   1. Extract CC from PTX `.target` (if provided)
 *   2. Probe nvidia-smi for device name + compute capability
 *   3. If SM count missing, try local CUDA detection (optional)
 *   4. If SM count still missing, infer from device name heuristics
 *   5. Fall back to DEFAULT_FALLBACK_CC (8.6 Ampere) if nothing found
 *
 * Priority for architecture:
 * PTX CC > nvidia-smi CC > DEFAULT_FALLBACK_CC
 *
 * Priority for SM count:
 * nvidia-smi > local CUDA detect > name heuristic > undefined
 *
 * @param presetKey Config value from user: "auto", "a100", "rtx-4090", etc.
 * @param options Optional: PTX text and local detection flag
 * @returns GpuSpec + resolution explanation string
 */
export async function resolveGpuSpecForAnalysis(
  presetKey: string,
  options: ResolveGpuSpecOptions = {}
): Promise<GpuSpecResolution> {
  const key = presetKey.trim().toLowerCase();
  
  // Handle named presets
  if (key && key !== "auto") {
    return {
      spec: getPreset(key),
      resolution: `preset:${key}`,
    };
  }

  // Auto mode: multi-source detection
  
  // 1. Extract CC from PTX if available
  const ptxCc = options.ptxText
    ? ccKeyFromPtxTarget(options.ptxText)
    : undefined;
  
  // 2. Query nvidia-smi for device + CC + SM count
  const probed = await queryFirstGpuFromNvidiaSmi();
  let { smCount, source: smCountSource } = resolveSmCountAndSource(probed);

  // 3. If SM count missing and local detection enabled, try CUDA detection
  if (smCount === undefined && options.enableLocalCudaDetect) {
    const local = await detectSmCountLocally({
      expectedName: probed.ok ? probed.row.name : undefined,
      expectedComputeCap: probed.ok ? probed.row.compute_cap : undefined,
    });
    if (local) {
      smCount = local.smCount;
      smCountSource = local.source;
    }
  }

  // 4. If SM count still missing, use name heuristic
  if (smCount === undefined && probed.ok) {
    const inferred = inferSmCountFromGpuName(probed.row.name);
    if (inferred !== undefined) {
      smCount = inferred;
      smCountSource = "name.heuristic";
    }
  }

  // 5. Determine final architecture and resolution string
  
  // Case A: PTX CC found + device detected via nvidia-smi
  if (ptxCc && probed.ok) {
    return {
      spec: gpuSpecFromArch(ptxCc, smCount),
      resolution: `ptx:${GPU_SM_CONFIGS[ptxCc].name} sm_count:${smCount ?? "?"} (${smCountSource}) device:${probed.row.name} (SMI CC ${probed.row.compute_cap})`,
    };
  }

  // Case B: PTX CC found but nvidia-smi failed
  if (ptxCc && !probed.ok) {
    const tag =
      probed.reason === "not_present"
        ? RESOLUTION_SMI_NOT_PRESENT
        : RESOLUTION_SMI_PROBE_FAILED;
    return {
      spec: gpuSpecFromArch(ptxCc, undefined),
      resolution: `ptx:${GPU_SM_CONFIGS[ptxCc].name} | ${tag}`,
    };
  }

  // Case C: nvidia-smi detected device but no PTX CC
  if (probed.ok) {
    const smiCc =
      computeCapToArchKey(probed.row.compute_cap) ?? DEFAULT_FALLBACK_CC;
    return {
      spec: gpuSpecFromArch(smiCc, smCount),
      resolution: `device:${probed.row.name} (CC ${probed.row.compute_cap}) → ${GPU_SM_CONFIGS[smiCc].name} sm_count:${smCount ?? "?"} (${smCountSource})`,
    };
  }

  // Case D: All sources failed - use fallback
  return {
    spec: gpuSpecFromArch(DEFAULT_FALLBACK_CC, undefined),
    resolution:
      probed.reason === "not_present"
        ? RESOLUTION_SMI_NOT_PRESENT
        : RESOLUTION_SMI_PROBE_FAILED,
  };
}

/**
 * Extracts SM version number from PTX `.target sm_XX` directive.
 *
 * The `.target` directive in PTX specifies minimum compute capability:
 * Example: ".target sm_80" (Ampere)
 * Parsed SM number: 80 -> "8.0" (via SM_VERSION_TO_CC table)
 *
 * Regex: /\.target\s+sm_(\d+)/i
 * Returns: captured numeric part, or undefined if not found
 *
 * @param ptx PTX IR text
 * @returns SM version number (e.g., 80), or undefined
 */
export function parsePtxSmTargetVersion(ptx: string): number | undefined {
  const m = /\.target\s+sm_(\d+)/i.exec(ptx);
  return m ? parseInt(m[1]!, 10) : undefined;
}
