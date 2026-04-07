/**
 * Parallel PTX per-`.entry` feature extraction (kernel-level parallelism).
 * Launch/occupancy still use the picked kernel via mergeLaunchWithHints (first or filter).
 * Pattern/memory/bottleneck consume summed instruction features across all entries.
 */

import { listPtxEntryNames } from "./ptx_parse";
import {
  emptyInstructionFeatures,
  extractInstructionFeaturesFromRange,
  extractOneKernelFeatures,
  findKernelBodySpan,
  sumPtxInstructionFeatures,
  type PtxInstructionFeatures,
} from "./ptx_features";
import { runWorkerScript } from "./worker_util";

/** Max concurrent kernel extract workers (parent awaits all; no env toggles). */
const KERNEL_EXTRACT_POOL_SIZE = 4;

async function mapPool<T, R>(
  items: T[],
  concurrency: number,
  mapper: (item: T, index: number) => Promise<R>
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;

  async function worker(): Promise<void> {
    while (true) {
      const idx = next++;
      if (idx >= items.length) {
        return;
      }
      results[idx] = await mapper(items[idx]!, idx);
    }
  }

  const nWorkers = Math.max(1, Math.min(concurrency, items.length));
  await Promise.all(Array.from({ length: nWorkers }, () => worker()));
  return results;
}

async function extractWithWorkers(
  ptx: string,
  names: string[]
): Promise<PtxInstructionFeatures[]> {
  return mapPool(names, KERNEL_EXTRACT_POOL_SIZE, (kernelSubstring) =>
    runWorkerScript("workers/ptxKernelExtractWorker.js", {
      ptx,
      kernelSubstring,
    }) as Promise<PtxInstructionFeatures>
  );
}

function extractSyncSequential(
  ptx: string,
  names: string[]
): PtxInstructionFeatures[] {
  return names.map((name) => extractOneKernelFeatures(ptx, name));
}

/**
 * When `kernelFilter` is set, only that kernel (same matching rules as merge).
 * Otherwise all `.entry` names are scanned; features are summed for downstream models.
 */
export async function extractPtxKernelsMerged(
  ptx: string,
  kernelFilter: string | undefined
): Promise<{ instr: PtxInstructionFeatures; kernelNames: string[] }> {
  if (kernelFilter !== undefined) {
    const span = findKernelBodySpan(ptx, kernelFilter);
    const instr = span
      ? extractInstructionFeaturesFromRange(
          ptx,
          span.bodyStart,
          span.bodyEnd
        )
      : emptyInstructionFeatures();
    return {
      instr,
      kernelNames: span ? [span.name] : [],
    };
  }

  const names = listPtxEntryNames(ptx);
  if (names.length === 0) {
    return { instr: emptyInstructionFeatures(), kernelNames: [] };
  }
  if (names.length === 1) {
    return {
      instr: extractOneKernelFeatures(ptx, undefined),
      kernelNames: names,
    };
  }

  let perKernel: PtxInstructionFeatures[];
  try {
    perKernel = await extractWithWorkers(ptx, names);
  } catch {
    perKernel = extractSyncSequential(ptx, names);
  }

  return {
    instr: sumPtxInstructionFeatures(perKernel),
    kernelNames: names,
  };
}
