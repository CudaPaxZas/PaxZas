/**
 * PTX instruction feature extraction (matches ptx_features.py).
 */

import { pickEntry } from "./ptx_parse";
import { forEachLineInRange } from "./line_scan";

export interface PtxInstructionFeatures {
  global_loads: number;
  global_stores: number;
  fma: number;
  add: number;
  mul: number;
  barrier: number;
  reg_decl_lines: number;
  branches: number;
  loops: number;
}

function emptyPtxFeatures(): PtxInstructionFeatures {
  return {
    global_loads: 0,
    global_stores: 0,
    fma: 0,
    add: 0,
    mul: 0,
    barrier: 0,
    reg_decl_lines: 0,
    branches: 0,
    loops: 0,
  };
}

const LD_GLOBAL = /\bld\.global\b/g;
const ST_GLOBAL = /\bst\.global\b/g;
const FMA = /\bfma\./g;
// Negative lookbehind prevents matching sub-opcodes in compound PTX instructions
// such as `red.add.s32` or `atom.add.f32`, which would inflate the FLOP count.
const ADD = /(?<![A-Za-z0-9_.])add\./g;
const MUL = /(?<![A-Za-z0-9_.])mul\./g;
const BARRIER = /\bbar\.sync\b/g;
const REG_LINE = /^\s*\.reg\b/;
const BRA_OPCODE = /\bbra(?:\.[A-Za-z0-9_]+)*\b/g;
const LABEL_DEF = /^\s*(\$?L[A-Za-z0-9_]+)\s*:/;
const BRA_TARGET = /\bbra(?:\.[A-Za-z0-9_]+)*\s+(\$?L[A-Za-z0-9_]+)\s*;/g;

export function ptxFeaturesAsDict(f: PtxInstructionFeatures): Record<string, number> {
  return {
    global_loads: f.global_loads,
    global_stores: f.global_stores,
    fma: f.fma,
    add: f.add,
    mul: f.mul,
    barrier: f.barrier,
    reg_decl_lines: f.reg_decl_lines,
    branches: f.branches,
    loops: f.loops,
  };
}

function stripPtxComment(line: string): string {
  const i = line.indexOf("//");
  if (i >= 0) {
    return line.slice(0, i);
  }
  return line;
}

export function extractBracedBody(
  ptx: string,
  openBraceIndex: number
): string | undefined {
  const span = extractBracedInnerRange(ptx, openBraceIndex);
  if (!span) {
    return undefined;
  }
  return ptx.slice(span[0], span[1]);
}

/** Inner kernel body as [start, end) indices into ptx (avoids allocating body substring for scan). */
export function extractBracedInnerRange(
  ptx: string,
  openBraceIndex: number
): [number, number] | undefined {
  let depth = 0;
  const n = ptx.length;
  for (let i = openBraceIndex; i < n; i++) {
    const c = ptx[i]!;
    if (c === "{") {
      depth++;
    } else if (c === "}") {
      depth--;
      if (depth === 0) {
        return [openBraceIndex + 1, i];
      }
    }
  }
  return undefined;
}

/**
 * Returns [kernel_name, body_text] for the chosen .entry, or [undefined, undefined].
 */
export function findKernelBody(
  ptx: string,
  kernelSubstring: string | undefined
): [string | undefined, string | undefined] {
  const picked = pickEntry(ptx, kernelSubstring);
  if (!picked) {
    return [undefined, undefined];
  }
  const { name, parenEnd } = picked;
  const chunk = ptx.slice(parenEnd);
  const brace = chunk.indexOf("{");
  if (brace === -1) {
    return [name, undefined];
  }
  const absBrace = parenEnd + brace;
  const body = extractBracedBody(ptx, absBrace);
  return [name, body];
}

/** Kernel body span [bodyStart, bodyEnd) into full PTX (no body string allocation). */
export function findKernelBodySpan(
  ptx: string,
  kernelSubstring: string | undefined
): { name: string; bodyStart: number; bodyEnd: number } | undefined {
  const picked = pickEntry(ptx, kernelSubstring);
  if (!picked) {
    return undefined;
  }
  const chunk = ptx.slice(picked.parenEnd);
  const brace = chunk.indexOf("{");
  if (brace === -1) {
    return undefined;
  }
  const absBrace = picked.parenEnd + brace;
  const span = extractBracedInnerRange(ptx, absBrace);
  if (!span) {
    return undefined;
  }
  return { name: picked.name, bodyStart: span[0], bodyEnd: span[1] };
}

function ptxBackwardBranchEdgesFromRange(
  ptx: string,
  bodyStart: number,
  bodyEnd: number
): number {
  const labelLine = new Map<string, number>();
  let lineNo = 0;
  forEachLineInRange(ptx, bodyStart, bodyEnd, (raw) => {
    const line = stripPtxComment(raw).trim();
    if (line) {
      const m = line.match(LABEL_DEF);
      if (m) {
        labelLine.set(m[1]!, lineNo);
      }
    }
    lineNo += 1;
  });
  let back = 0;
  lineNo = 0;
  forEachLineInRange(ptx, bodyStart, bodyEnd, (raw) => {
    const line = stripPtxComment(raw).trim();
    if (line) {
      BRA_TARGET.lastIndex = 0;
      let bm: RegExpExecArray | null;
      while ((bm = BRA_TARGET.exec(line)) !== null) {
        const lab = bm[1]!;
        const li = labelLine.get(lab);
        if (li !== undefined && li < lineNo) {
          back++;
        }
      }
    }
    lineNo += 1;
  });
  return back;
}

export function extractInstructionFeaturesFromRange(
  ptx: string,
  bodyStart: number,
  bodyEnd: number
): PtxInstructionFeatures {
  const f = emptyPtxFeatures();
  forEachLineInRange(ptx, bodyStart, bodyEnd, (raw) => {
    const line = stripPtxComment(raw).trim();
    if (!line) {
      return;
    }
    if (REG_LINE.test(line)) {
      f.reg_decl_lines += 1;
    }
    LD_GLOBAL.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = LD_GLOBAL.exec(line)) !== null) {
      f.global_loads += 1;
    }
    ST_GLOBAL.lastIndex = 0;
    while ((m = ST_GLOBAL.exec(line)) !== null) {
      f.global_stores += 1;
    }
    FMA.lastIndex = 0;
    while ((m = FMA.exec(line)) !== null) {
      f.fma += 1;
    }
    ADD.lastIndex = 0;
    while ((m = ADD.exec(line)) !== null) {
      f.add += 1;
    }
    MUL.lastIndex = 0;
    while ((m = MUL.exec(line)) !== null) {
      f.mul += 1;
    }
    BARRIER.lastIndex = 0;
    while ((m = BARRIER.exec(line)) !== null) {
      f.barrier += 1;
    }
    BRA_OPCODE.lastIndex = 0;
    while ((m = BRA_OPCODE.exec(line)) !== null) {
      f.branches += 1;
    }
  });
  f.loops = ptxBackwardBranchEdgesFromRange(ptx, bodyStart, bodyEnd);
  return f;
}

export function extractInstructionFeatures(
  kernelBody: string
): PtxInstructionFeatures {
  if (!kernelBody) {
    return emptyPtxFeatures();
  }
  return extractInstructionFeaturesFromRange(kernelBody, 0, kernelBody.length);
}

export function flopsHeuristic(f: PtxInstructionFeatures): number {
  return f.fma * 2 + f.add + f.mul;
}

export function bytesHeuristic(
  f: PtxInstructionFeatures,
  bytesPerMemOp = 4
): number {
  return (f.global_loads + f.global_stores) * bytesPerMemOp;
}

export function buildFeatureBundle(
  registersPerThread: number,
  instr: PtxInstructionFeatures
): Record<string, number> {
  return { ...ptxFeaturesAsDict(instr), registers: registersPerThread };
}

export function emptyInstructionFeatures(): PtxInstructionFeatures {
  return emptyPtxFeatures();
}

/** Sum instruction counters across kernels (whole-module proxy; `loops` is a heuristic sum). */
export function sumPtxInstructionFeatures(
  parts: PtxInstructionFeatures[]
): PtxInstructionFeatures {
  const out = emptyPtxFeatures();
  for (const p of parts) {
    out.global_loads += p.global_loads;
    out.global_stores += p.global_stores;
    out.fma += p.fma;
    out.add += p.add;
    out.mul += p.mul;
    out.barrier += p.barrier;
    out.reg_decl_lines += p.reg_decl_lines;
    out.branches += p.branches;
    out.loops += p.loops;
  }
  return out;
}

/** Features for one `.entry` (by exact / partial name, or first when undefined). */
export function extractOneKernelFeatures(
  ptx: string,
  kernelSubstring: string | undefined
): PtxInstructionFeatures {
  const span = findKernelBodySpan(ptx, kernelSubstring);
  if (!span) {
    return emptyPtxFeatures();
  }
  return extractInstructionFeaturesFromRange(
    ptx,
    span.bodyStart,
    span.bodyEnd
  );
}
