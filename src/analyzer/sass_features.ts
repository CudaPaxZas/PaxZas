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
  /**
   * STG.128 / STG.E.128: 128-bit (16-byte) global stores.
   * Mirrors ldg_128 on the write side — used in estimateSassBytes() to give
   * precise store bandwidth instead of the previous 4-byte-per-store assumption.
   */
  stg_128: number;
  /**
   * STG.64 / STG.E.64: 64-bit (8-byte) global stores.
   */
  stg_64: number;
  /**
   * STG.32 / STG.E.32: 32-bit (4-byte) global stores.
   * Most common CUDA store width for fp32 element-wise kernels.
   */
  stg_32: number;
  cg_loads: number;
  cs_loads: number;
  shared_loads: number;
  shared_stores: number;
  arithmetic_ops: number;
  /** Integer ALU/control ops used to separate FP-vs-address compute mix. */
  integer_ops: number;
  tensor_ops: number;
  barrier: number;
  branch: number;
  /** LDL: local-memory loads — evidence that the compiler spilled registers to local memory. */
  local_loads: number;
  /** STL: local-memory stores — the write half of a register-file spill. */
  local_stores: number;
  /** HMMA / WGMMA / WMMA instructions — floating-point tensor-core ops only (excludes IMMA/BMMA). */
  wmma_ops: number;
  /**
   * ATOM / ATOMS / RED instructions — total atomic operation count.
   *
   * Includes ALL serializing atomic instructions:
   *   - ATOM.*  — global-memory read-modify-write (serialized via L2 partition)
   *   - RED.*   — global fire-and-forget reduction (also serialized at L2)
   *   - ATOMS.* — shared-memory atomic (serialized at shared-memory bank level;
   *                fast, no L2 contention)
   *
   * For L2 contention analysis use `global_atomic_ops` (ATOM + RED only).
   */
  atomic_ops: number;
  /**
   * ATOM / RED instructions only — global-memory atomics that serialize through
   * the L2 cache partition.
   *
   * Excludes ATOMS (shared-memory atomics), which cause shared-bank contention
   * rather than L2 serialisation.  This is the correct numerator for the
   * `atomic_contention_risk` pattern signal.
   */
  global_atomic_ops: number;
  /**
   * SHFL.* instructions — warp shuffle / register-exchange operations.
   *
   * Shuffle runs entirely within the warp's register file (no shared-memory
   * bandwidth, no barriers required).  A reduction or broadcast built on
   * SHFL is 2–4× faster than the equivalent LDS + BAR pattern.
   * Presence is a positive quality signal in `reduction`-classified kernels;
   * absence in a high-barrier reduction is a missed-opportunity flag (F4).
   */
  warp_shuffle_ops: number;
  /**
   * VOTE.* / MATCH.* instructions — warp-vote and warp-match predicates.
   *
   * Warp-vote ops (VOTE.ALL, VOTE.ANY, VOTE.EQ, MATCH.ANY) evaluate a boolean
   * across all 32 active threads and return a lane-mask.  Heavy use indicates
   * predicate-driven divergence management — a positive sign in
   * `control_heavy` kernels because it means the programmer is at least
   * coalescing branching decisions at warp granularity.
   */
  warp_vote_ops: number;
  /**
   * MUFU.* instructions — Special Function Unit operations.
   *
   * The SFU pipeline has ¼ the throughput of the main ALU (one result per
   * 4 clock cycles per warp vs. one per cycle for FFMA).  Transcendental
   * functions compiled by CUDA — sinf, cosf, expf, logf, rcpf, rsqrtf,
   * sqrtf — all lower to MUFU.  An activation-heavy kernel (sigmoid, GELU,
   * softmax) with many MUFU instructions may be SFU-bound rather than
   * memory- or CUDA-core-bound, a bottleneck invisible to arithmetic-intensity
   * analysis alone.
   */
  sfu_ops: number;
  /**
   * Scalar FP16 arithmetic instructions — HFMA, HADD, HMUL.
   *
   * These run on CUDA cores at 2× the throughput of FP32, but they are
   * still scalar lanes.  On SM ≥ 7.0 (Volta+) the same FP16 matrix work
   * expressed as WMMA / HMMA tensor-core operations achieves 16–32× higher
   * throughput.  A kernel with heavy fp16_arith_ops but zero wmma_ops is
   * leaving substantial performance on the table.
   *
   * NOTE: fp16_arith_ops are also counted in arithmetic_ops for totals; this
   * field is a subset that isolates the FP16 scalar path specifically.
   */
  fp16_arith_ops: number;
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
    stg_128: 0,
    stg_64: 0,
    stg_32: 0,
    cg_loads: 0,
    cs_loads: 0,
    shared_loads: 0,
    shared_stores: 0,
    arithmetic_ops: 0,
    integer_ops: 0,
    tensor_ops: 0,
    barrier: 0,
    branch: 0,
    local_loads: 0,
    local_stores: 0,
    wmma_ops: 0,
    atomic_ops: 0,
    global_atomic_ops: 0,
    warp_shuffle_ops: 0,
    warp_vote_ops: 0,
    sfu_ops: 0,
    fp16_arith_ops: 0,
    total_instructions: 0,
    max_register_index: -1,
    instruction_sequence: [],
    stream_interleave_score: 0,
    stream_max_consecutive_loads: 0,
  };
}

const FUNCTION_RE = /^\s*Function\s*:\s*(.+?)\s*$/;
const SASS_SM_TARGET_RE = /\b(?:code\s+for\s+sm_|arch\s*=\s*sm_|sm_)(\d{2,3})\b/gi;
/**
 * Matches the opcode from a cuobjdump SASS disassembly line.
 * The non-capturing group `(?:@!?P\d+\s+)` strips predicate guards such as
 * `@P0`, `@!P1`, etc. so that predicated instructions like `@P0 BRA 0x200`
 * are correctly extracted as `BRA` rather than missed entirely.
 */
const SASS_OPCODE_RE = /\/\*[0-9A-Fa-f]+\*\/\s*(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)/;
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
    // Mirror the LDG byte-width tracking for stores so that estimateSassBytes()
    // can compute write bandwidth accurately.  Without this, all stores were
    // assumed 4 bytes, causing a 4× underestimate of write bandwidth on kernels
    // that emit STG.E.128 (16-byte coalesced stores).
    if (opcode.includes(".128")) {
      f.stg_128 += 1;
    } else if (
      opcode.includes(".64") ||
      opcode.includes(".U64") ||
      opcode.includes(".S64") ||
      opcode.includes(".F64")
    ) {
      f.stg_64 += 1;
    } else if (
      opcode.includes(".32") ||
      opcode.includes(".U32") ||
      opcode.includes(".S32") ||
      opcode.includes(".F32")
    ) {
      f.stg_32 += 1;
    }
  } else if (opcode.startsWith("LDS")) {
    f.shared_loads += 1;
  } else if (opcode.startsWith("STS")) {
    f.shared_stores += 1;
  } else if (opcode.startsWith("LDL")) {
    // Local-memory load: restores a value that the compiler spilled from the
    // register file into per-thread local memory (backed by L1/L2/DRAM).
    // Even a few LDL per warp thread can stall execution for 100+ cycles.
    f.local_loads += 1;
  } else if (opcode.startsWith("STL")) {
    // Local-memory store: the spill write that precedes an LDL reload.
    f.local_stores += 1;
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

  if (
    startsWithAny(opcode, [
      "IADD",
      "IMAD",
      "IMUL",
      "IMNMX",
      "ISCADD",
      "ISET",
      "ICMP",
      "IABS",
      "INEG",
      "IAND",
      "IOR",
      "IXOR",
      "ISHL",
      "ISHR",
    ])
  ) {
    f.integer_ops += 1;
  }

  if (startsWithAny(opcode, ["MMA", "HMMA", "IMMA", "BMMA", "WGMMA", "WMMA"])) {
    f.tensor_ops += 1;
    // wmma_ops counts only FP matrix-unit instructions:
    //   HMMA  — Volta/Ampere hardware WMMA (fragment API compiles to this)
    //   WGMMA — Hopper warp-group MMA (SM90+)
    //   WMMA  — explicit WMMA opcode emitted by some disassemblers
    // Integer (IMMA) and binary (BMMA) tensor ops are excluded.
    if (startsWithAny(opcode, ["HMMA", "WGMMA", "WMMA"])) {
      f.wmma_ops += 1;
    }
  }

  // Atomic memory operations:
  //   ATOMS.*  — shared-memory atomic; fast bank-level serialisation only.
  //              Counted in atomic_ops (total) but NOT in global_atomic_ops.
  //   ATOM.*   — global-memory atomic; serialized through L2 partition.
  //   RED.*    — global fire-and-forget reduction; also serialized at L2.
  // Check ATOMS before ATOM because "ATOMS".startsWith("ATOM") === true.
  if (opcode.startsWith("ATOMS")) {
    f.atomic_ops += 1;           // shared atomic: counted in total, not global
  } else if (startsWithAny(opcode, ["ATOM", "RED"])) {
    f.atomic_ops += 1;           // global atomic: counted in both
    f.global_atomic_ops += 1;
  }

  // Warp shuffle — SHFL.BFLY / SHFL.UP / SHFL.DOWN / SHFL.IDX.
  // All variants start with "SHFL"; they execute in the register file (no
  // shared-memory traffic, no barrier required).
  if (opcode.startsWith("SHFL")) {
    f.warp_shuffle_ops += 1;
  }

  // Warp-vote / warp-match — VOTE.ALL / VOTE.ANY / VOTE.EQ / MATCH.ANY / MATCH.ALL.
  // Returns a lane-mask; used for predicate coalescing and cooperative decisions.
  if (opcode.startsWith("VOTE") || opcode.startsWith("MATCH")) {
    f.warp_vote_ops += 1;
  }

  // Special Function Unit: MUFU dispatches to the ¼-throughput transcendental
  // pipeline.  All variants (SIN, COS, EXP2, LOG2, RCP, RSQ, SQRT, etc.) are
  // counted together because they all compete for the same hardware resource.
  if (opcode.startsWith("MUFU")) {
    f.sfu_ops += 1;
  }

  // Scalar FP16 arithmetic — a subset of arithmetic_ops.
  // HFMA = half-precision FMA, HADD = half-precision add, HMUL = half-precision mul.
  // Counted here in addition to arithmetic_ops so the pattern model can ask
  // "is this an FP16 scalar kernel that should be using tensor cores instead?"
  if (startsWithAny(opcode, ["HFMA", "HADD", "HMUL"])) {
    f.fp16_arith_ops += 1;
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
    stg_128: f.stg_128,
    stg_64: f.stg_64,
    stg_32: f.stg_32,
    cg_loads: f.cg_loads,
    cs_loads: f.cs_loads,
    shared_loads: f.shared_loads,
    shared_stores: f.shared_stores,
    arithmetic_ops: f.arithmetic_ops,
    integer_ops: f.integer_ops,
    tensor_ops: f.tensor_ops,
    barrier: f.barrier,
    branch: f.branch,
    local_loads: f.local_loads,
    local_stores: f.local_stores,
    wmma_ops: f.wmma_ops,
    atomic_ops: f.atomic_ops,
    global_atomic_ops: f.global_atomic_ops,
    warp_shuffle_ops: f.warp_shuffle_ops,
    warp_vote_ops: f.warp_vote_ops,
    sfu_ops: f.sfu_ops,
    fp16_arith_ops: f.fp16_arith_ops,
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

export function detectSassSmTargets(sassText: string): number[] {
  const out = new Set<number>();
  let m: RegExpExecArray | null;
  SASS_SM_TARGET_RE.lastIndex = 0;
  while ((m = SASS_SM_TARGET_RE.exec(sassText)) !== null) {
    const sm = parseInt(m[1]!, 10);
    if (Number.isFinite(sm) && sm > 0) {
      out.add(sm);
    }
  }
  return Array.from(out.values()).sort((a, b) => a - b);
}
