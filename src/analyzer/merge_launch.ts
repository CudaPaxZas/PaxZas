/**
 * Merge launch dict with PTX hints (matches pipeline._merge_launch_with_hints).
 */

import { maxThreadsFromMaxntid, parsePtxKernelHints, type PtxKernelHints } from "./ptx_parse";

export interface MergeLaunchResult {
  threads: number;
  shared: number;
  registers: number;
  /** Optional total thread blocks in the launch (I3 — SM util cap). */
  gridBlocks?: number;
  hints: PtxKernelHints;
  registerSource: string;
  threadsSource: string;
  sharedSource: string;
}

export function mergeLaunchWithHints(
  ptx: string,
  kernelSubstring: string | undefined,
  launch: Partial<Record<"threads" | "shared" | "registers" | "grid", number>>,
  sassRegisters: number | null | undefined
): MergeLaunchResult {
  const hints = parsePtxKernelHints(ptx, kernelSubstring);
  if (hints.kernelName === undefined) {
    throw new Error("No .entry kernel matched; use --kernel or check PTX.");
  }

  let registerSource = "launch";
  let registers = launch.registers;
  if (registers === undefined) {
    registers = hints.maxnreg;
    registerSource = "ptx.maxnreg";
  }
  if (registers === undefined && sassRegisters != null) {
    registers = sassRegisters;
    registerSource = "sass.inferred";
  }
  if (registers === undefined) {
    throw new Error(
      "Need regs= in --launch, .maxnreg in PTX, or inferable SASS registers."
    );
  }

  let threadsSource = "launch";
  let threads = launch.threads;
  if (threads === undefined) {
    const m = maxThreadsFromMaxntid(hints.maxntid);
    threads = m;
    threadsSource = "ptx.maxntid";
  }
  if (threads === undefined) {
    throw new Error("Need threads= in --launch or .maxntid in PTX.");
  }

  let sharedSource = "launch";
  let shared: number;
  if (launch.shared !== undefined) {
    shared = launch.shared;
  } else {
    shared = hints.staticSharedBytes;
    sharedSource = "ptx.static_shared";
  }

  const gridBlocks =
    launch.grid !== undefined ? Math.floor(launch.grid) : undefined;

  return {
    threads: Math.floor(threads),
    shared: Math.floor(shared),
    registers: Math.floor(registers),
    gridBlocks,
    hints,
    registerSource,
    threadsSource,
    sharedSource,
  };
}
