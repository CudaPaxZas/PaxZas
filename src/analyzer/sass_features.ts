/**
 * SASS dump feature extraction (matches sass_features.py).
 * Large-file: single forward scan, first (or matched) section only; bounded opcode list.
 */

import {
  createInterleaveState,
  interleaveScoreFromState,
  observeSassOpcodeForPatternMetrics,
} from "./opcode_kinds";

export interface SassInstructionFeatures {
  global_loads: number;
  global_stores: number;
  ldg_128: number;
  ldg_64: number;
  ldg_32: number;
  cg_loads: number;
  cs_loads: number;
  shared_loads: number;
  shared_stores: number;
  arithmetic_ops: number;
  tensor_ops: number;
  barrier: number;
  branch: number;
  total_instructions: number;
  max_register_index: number;
  instruction_sequence: string[];
  /** Streaming metrics (same semantics as pattern_model on full sequence). */
  stream_interleave_score: number;
  stream_max_consecutive_loads: number;
}

const MAX_INSTRUCTION_SEQUENCE = 65536;

function emptySass(): SassInstructionFeatures {
  return {
    global_loads: 0,
    global_stores: 0,
    ldg_128: 0,
    ldg_64: 0,
    ldg_32: 0,
    cg_loads: 0,
    cs_loads: 0,
    shared_loads: 0,
    shared_stores: 0,
    arithmetic_ops: 0,
    tensor_ops: 0,
    barrier: 0,
    branch: 0,
    total_instructions: 0,
    max_register_index: -1,
    instruction_sequence: [],
    stream_interleave_score: 0,
    stream_max_consecutive_loads: 0,
  };
}

const FUNCTION_RE = /^\s*Function\s*:\s*(.+?)\s*$/;
const SASS_OPCODE_RE = /\/\*[0-9A-Fa-f]+\*\/\s*([A-Z][A-Z0-9_.]+)/;
const REGISTER_RE = /\bR(\d+)\b/g;

function startsWithAny(opcode: string, prefixes: string[]): boolean {
  for (const p of prefixes) {
    if (opcode.startsWith(p)) {
      return true;
    }
  }
  return false;
}

function classifyOpcode(op: string, f: SassInstructionFeatures): void {
  const opcode = op.toUpperCase();
  f.total_instructions += 1;

  if (opcode.startsWith("LDG")) {
    f.global_loads += 1;
    if (opcode.includes(".128")) {
      f.ldg_128 += 1;
    } else if (
      opcode.includes(".64") ||
      opcode.includes(".U64") ||
      opcode.includes(".S64") ||
      opcode.includes(".F64")
    ) {
      f.ldg_64 += 1;
    } else if (
      opcode.includes(".32") ||
      opcode.includes(".U32") ||
      opcode.includes(".S32") ||
      opcode.includes(".F32")
    ) {
      f.ldg_32 += 1;
    }
    if (opcode.includes(".CG")) {
      f.cg_loads += 1;
    }
    if (opcode.includes(".CS")) {
      f.cs_loads += 1;
    }
  } else if (opcode.startsWith("STG")) {
    f.global_stores += 1;
  } else if (opcode.startsWith("LDS")) {
    f.shared_loads += 1;
  } else if (opcode.startsWith("STS")) {
    f.shared_stores += 1;
  }

  if (
    startsWithAny(opcode, [
      "FFMA",
      "FADD",
      "FMUL",
      "IADD",
      "IMAD",
      "IMUL",
      "HFMA",
      "HADD",
      "HMUL",
    ])
  ) {
    f.arithmetic_ops += 1;
  }

  if (startsWithAny(opcode, ["MMA", "HMMA", "IMMA", "BMMA", "WGMMA"])) {
    f.tensor_ops += 1;
  }

  if (startsWithAny(opcode, ["BAR", "DEPBAR", "MEMBAR"])) {
    f.barrier += 1;
  }

  if (startsWithAny(opcode, ["BRA", "JMP", "RET", "SSY", "SYNC"])) {
    f.branch += 1;
  }
}

function updateRegisterUsage(line: string, f: SassInstructionFeatures): void {
  REGISTER_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = REGISTER_RE.exec(line)) !== null) {
    const idx = parseInt(m[1]!, 10);
    if (idx > f.max_register_index) {
      f.max_register_index = idx;
    }
  }
}

/**
 * Single-pass scan: only the first `Function:` section (or the one matching kernelSubstring).
 */
export function extractSassFeatures(
  sassText: string,
  kernelSubstring: string | undefined
): [string | undefined, SassInstructionFeatures] {
  const f = emptySass();
  const interState = createInterleaveState();

  let collecting = false;
  let matchedName: string | undefined;
  let scanDone = false;

  const n = sassText.length;
  let i = 0;

  while (i < n && !scanDone) {
    const nl = sassText.indexOf("\n", i);
    const lineEnd = nl === -1 ? n : nl;
    let line = sassText.slice(i, lineEnd);
    if (line.charCodeAt(line.length - 1) === 13) {
      line = line.slice(0, -1);
    }

    const fm = line.match(FUNCTION_RE);
    if (fm) {
      const name = fm[1]!.trim();
      if (collecting) {
        scanDone = true;
        collecting = false;
        break;
      }
      if (kernelSubstring === undefined) {
        matchedName = name;
        collecting = true;
      } else if (name === kernelSubstring || name.includes(kernelSubstring)) {
        matchedName = name;
        collecting = true;
      }
    } else if (collecting) {
      const m = SASS_OPCODE_RE.exec(line);
      if (m) {
        const op = m[1]!;
        classifyOpcode(op, f);
        observeSassOpcodeForPatternMetrics(op, interState);
        if (f.instruction_sequence.length < MAX_INSTRUCTION_SEQUENCE) {
          f.instruction_sequence.push(op);
        }
      }
      updateRegisterUsage(line, f);
    }

    if (nl === -1) {
      break;
    }
    i = nl + 1;
  }

  f.stream_interleave_score = interleaveScoreFromState(interState);
  f.stream_max_consecutive_loads = interState.maxLoadRun;

  return [matchedName, f];
}

export function sassFeaturesAsDict(
  f: SassInstructionFeatures
): Record<string, number> {
  return {
    global_loads: f.global_loads,
    global_stores: f.global_stores,
    ldg_128: f.ldg_128,
    ldg_64: f.ldg_64,
    ldg_32: f.ldg_32,
    cg_loads: f.cg_loads,
    cs_loads: f.cs_loads,
    shared_loads: f.shared_loads,
    shared_stores: f.shared_stores,
    arithmetic_ops: f.arithmetic_ops,
    tensor_ops: f.tensor_ops,
    barrier: f.barrier,
    branch: f.branch,
    total_instructions: f.total_instructions,
    max_register_index: f.max_register_index,
  };
}

export function buildSassFeatureBundle(
  registersPerThread: number,
  instr: SassInstructionFeatures
): Record<string, number | null> {
  return {
    ...sassFeaturesAsDict(instr),
    registers: registersPerThread,
    inferred_registers_per_thread:
      instr.max_register_index >= 0 ? instr.max_register_index + 1 : null,
  };
}

export function inferRegistersPerThreadFromSass(
  sassText: string,
  kernelSubstring: string | undefined
): number | undefined {
  const [, instr] = extractSassFeatures(sassText, kernelSubstring);
  if (instr.max_register_index < 0) {
    return undefined;
  }
  return instr.max_register_index + 1;
}
