/** Per-SM limits for occupancy heuristics: architecture tables + device SM count from nvidia-smi. */

import {
  queryFirstGpuFromNvidiaSmi,
  RESOLUTION_SMI_NOT_PRESENT,
  RESOLUTION_SMI_PROBE_FAILED,
} from "./nvidia_smi";

export interface GpuSpec {
  name: string;
  smMaxThreads: number;
  smMaxBlocks: number;
  smMaxSharedMem: number;
  smMaxRegisters: number;
  warpSize: number;
  smMaxWarps: number | undefined;
  regAllocUnitPerWarp: number;
  sharedMemAllocUnit: number;
  minSharedPerBlockAlloc: number;
  smCount: number | undefined;
}

export type GpuArchLimits = Omit<GpuSpec, "smCount">;

export type GpuComputeCapabilityKey =
  | "7.0"
  | "7.5"
  | "8.0"
  | "8.6"
  | "8.9"
  | "9.0"
  | "10.0";

/** Per compute-capability SM limits (PTX .target / SMI compute_cap). smCount comes from nvidia-smi. */
export const GPU_SM_CONFIGS: Record<GpuComputeCapabilityKey, GpuArchLimits> = {
  "7.0": {
    name: "volta-sm70",
    smMaxThreads: 2048,
    smMaxWarps: 64,
    warpSize: 32,
    smMaxBlocks: 32,
    smMaxRegisters: 65536,
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
    regAllocUnitPerWarp: 256,
    smMaxSharedMem: 228 * 1024,
    sharedMemAllocUnit: 256,
    minSharedPerBlockAlloc: 256,
  },
};

export const GPU_COMPUTE_CAPABILITY_KEYS = Object.keys(
  GPU_SM_CONFIGS
) as GpuComputeCapabilityKey[];

/** When PTX and SMI are both missing or unusable for CC. */
export const DEFAULT_FALLBACK_CC: GpuComputeCapabilityKey = "8.6";

export const AMPERE_LIKE_DEFAULT: GpuSpec = {
  ...GPU_SM_CONFIGS[DEFAULT_FALLBACK_CC],
  smCount: undefined,
};

const PRESET_ARCH: Record<string, GpuComputeCapabilityKey> = {
  "ampere-like-default": "8.6",
  a100: "8.0",
  "rtx-4090": "8.9",
};

const PRESET_SM_COUNT: Record<string, number | undefined> = {
  "ampere-like-default": undefined,
  a100: 108,
  "rtx-4090": 128,
};

export const KNOWN_PRESET_KEYS = Object.keys(PRESET_ARCH).sort() as string[];

export function getPreset(name: string): GpuSpec {
  const key = name.trim().toLowerCase();
  const cc = PRESET_ARCH[key];
  if (!cc) {
    throw new Error(
      `Unknown GPU preset ${JSON.stringify(name)}; known: ${KNOWN_PRESET_KEYS.join(", ")}`
    );
  }
  return gpuSpecFromArch(cc, PRESET_SM_COUNT[key]);
}

export function gpuSpecFromArch(
  cc: GpuComputeCapabilityKey,
  smCount: number | undefined
): GpuSpec {
  return { ...GPU_SM_CONFIGS[cc], smCount: smCount ?? undefined };
}

/** Map PTX `.target sm_XX` to our CC key (undefined if unknown). */
export function parsePtxSmTargetVersion(ptx: string): number | undefined {
  const m = /\.target\s+sm_(\d+)/i.exec(ptx);
  return m ? parseInt(m[1]!, 10) : undefined;
}

const SM_VERSION_TO_CC: Partial<Record<number, GpuComputeCapabilityKey>> = {
  70: "7.0",
  75: "7.5",
  80: "8.0",
  86: "8.6",
  87: "8.6",
  89: "8.9",
  90: "9.0",
  /** CC 10.0 datacenter Blackwell; use SMI/CC for 10.3 / 12.x PTX targets not listed here. */
  100: "10.0",
};

export function ccKeyFromPtxTarget(ptx: string): GpuComputeCapabilityKey | undefined {
  const sm = parsePtxSmTargetVersion(ptx);
  if (sm === undefined) {
    return undefined;
  }
  return SM_VERSION_TO_CC[sm];
}

/** Normalize SMI `compute_cap` string to a table key. */
export function computeCapToArchKey(s: string): GpuComputeCapabilityKey | undefined {
  const n = parseFloat(s.trim());
  if (!Number.isFinite(n)) {
    return undefined;
  }
  const k = n.toFixed(1) as GpuComputeCapabilityKey;
  return k in GPU_SM_CONFIGS ? k : undefined;
}

export interface GpuSpecResolution {
  spec: GpuSpec;
  resolution: string;
}

export interface ResolveGpuSpecOptions {
  /** When set and contains `.target sm_XX`, drives arch table; sm_count still from SMI in auto mode. */
  ptxText?: string;
}

/**
 * @param presetKey - VS Code: `auto` uses PTX CC + GPU_SM_CONFIGS and nvidia-smi for sm_count; else named preset.
 */
export async function resolveGpuSpecForAnalysis(
  presetKey: string,
  options: ResolveGpuSpecOptions = {}
): Promise<GpuSpecResolution> {
  const key = presetKey.trim().toLowerCase();
  if (key && key !== "auto") {
    return {
      spec: getPreset(key),
      resolution: `preset:${key}`,
    };
  }

  const ptxCc = options.ptxText
    ? ccKeyFromPtxTarget(options.ptxText)
    : undefined;
  const probed = await queryFirstGpuFromNvidiaSmi();
  const smCount =
    probed.ok && probed.row.multiprocessor_count != null
      ? probed.row.multiprocessor_count
      : undefined;

  if (ptxCc && probed.ok) {
    return {
      spec: gpuSpecFromArch(ptxCc, smCount),
      resolution: `ptx:${GPU_SM_CONFIGS[ptxCc].name} sm_count:${smCount ?? "?"} device:${probed.row.name} (SMI CC ${probed.row.compute_cap})`,
    };
  }

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

  if (probed.ok) {
    const smiCc =
      computeCapToArchKey(probed.row.compute_cap) ?? DEFAULT_FALLBACK_CC;
    return {
      spec: gpuSpecFromArch(smiCc, smCount),
      resolution: `device:${probed.row.name} (CC ${probed.row.compute_cap}) → ${GPU_SM_CONFIGS[smiCc].name} sm_count:${smCount ?? "?"}`,
    };
  }

  return {
    spec: gpuSpecFromArch(DEFAULT_FALLBACK_CC, undefined),
    resolution:
      probed.reason === "not_present"
        ? RESOLUTION_SMI_NOT_PRESENT
        : RESOLUTION_SMI_PROBE_FAILED,
  };
}
