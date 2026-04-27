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
   * Gap 2: 16-bit (2-byte) global loads — LDG.E.U16 / .S16 / .F16 / .16.
   *
   * Without this counter, `half`/`bf16`/`int16` loads fell into the
   * unknown-width fallback (4 bytes), inflating estimated bandwidth by 2×
   * and pushing arithmetic intensity below thresholds for FP16 kernels.
   */
  ldg_16: number;
  /**
   * Gap 2: 8-bit (1-byte) global loads — LDG.E.U8 / .S8 / .8.
   *
   * Used by quantised inference (INT8 / FP8) and any `char`-array kernel.
   * Without explicit width tracking the bandwidth estimate was 4× too high.
   */
  ldg_8: number;
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
  /** Gap 2: 16-bit global stores — STG.16 / STG.E.U16 / .S16 / .F16. */
  stg_16: number;
  /** Gap 2: 8-bit global stores — STG.8 / STG.E.U8 / .S8. */
  stg_8: number;
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
  /**
   * Gap 7: kernel exit/return instructions (RET, EXIT) split out from `branch`.
   *
   * RET / EXIT fire at most a handful of times per kernel (one per early-return
   * path).  Folding them into the generic `branch` counter inflated branch
   * density on small kernels and made `branch_per_global_mem_op` noisy.
   * Counting them separately preserves the exit count while letting `branch`
   * reflect only true control-flow opcodes (BRA, JMP, SSY, SYNC).
   */
  kernel_exit: number;
  /**
   * Gap 8: SASS back-edges — `BRA <addr>` instructions whose target address
   * is *less than* the issuing PC (i.e. a backward jump that closes a loop).
   *
   * The compiler frequently fully-unrolls outer loops while keeping a sync
   * inside the body; in those cases `loops` (PTX-derived) is 0 even though
   * the kernel still issues one barrier per logical iteration.  Counting
   * SASS back-edges lets `over_synchronized` fire on fully-unrolled cases.
   */
  back_edges: number;
  /**
   * Gap 2 / G3: constant-memory loads — `LDC.*`.
   *
   * LDC pulls from the per-kernel constant bank (broadcast, cached at L1).
   * It is not global memory but is a real memory operation; counting it
   * separately keeps `global_loads` honest while still letting downstream
   * analyses observe constant-bank traffic.
   */
  const_loads: number;
  /**
   * Gap 3: TMA (Tensor Memory Accelerator) operations — `UTMALDG`, `UTMASTG`.
   *
   * Hopper / Blackwell async bulk copy.  These are real global-memory ops
   * (counted in global_loads / global_stores below) but also tracked here
   * so the pattern model can detect TMA-driven kernels.
   */
  tma_ops: number;
  /**
   * Gap 2 / G3: async global→shared copy — `LDGSTS.*` (Ampere) / `CP.ASYNC.*`
   * (Hopper/Blackwell).  These are global loads that bypass the register file
   * and stream directly into shared memory.  Counted in `global_loads`; this
   * field exposes the async sub-share for pattern analysis.
   */
  async_global_loads: number;
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
  /**
   * Gap 1: scalar FP64 arithmetic — DFMA / DADD / DMUL / DMNMX / DSETP.
   *
   * Double-precision throughput on most consumer GPUs is 1/32 of FP32, but
   * before this counter existed FP64 ops were entirely invisible — a CFD or
   * MD kernel showed `compute_ops ≈ 0` and was always classified as
   * `memory_bound`.  fp64_arith_ops is *also* counted in `arithmetic_ops`
   * (so totals stay consistent); this field isolates the FP64 sub-share so
   * the FLOP proxy can weight it correctly.
   */
  fp64_arith_ops: number;
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
    ldg_16: 0,
    ldg_8: 0,
    stg_128: 0,
    stg_64: 0,
    stg_32: 0,
    stg_16: 0,
    stg_8: 0,
    cg_loads: 0,
    cs_loads: 0,
    shared_loads: 0,
    shared_stores: 0,
    arithmetic_ops: 0,
    integer_ops: 0,
    tensor_ops: 0,
    barrier: 0,
    branch: 0,
    kernel_exit: 0,
    back_edges: 0,
    const_loads: 0,
    tma_ops: 0,
    async_global_loads: 0,
    local_loads: 0,
    local_stores: 0,
    wmma_ops: 0,
    atomic_ops: 0,
    global_atomic_ops: 0,
    warp_shuffle_ops: 0,
    warp_vote_ops: 0,
    sfu_ops: 0,
    fp16_arith_ops: 0,
    fp64_arith_ops: 0,
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
 *
 * Capture group 1: address comment (hex); group 2: opcode.
 */
const SASS_OPCODE_RE = /\/\*([0-9A-Fa-f]+)\*\/\s*(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)/;
const REGISTER_RE = /\bR(\d+)\b/g;
/** Captures a hex branch target like `0x1ff0` from a BRA/JMP operand. */
const BRA_TARGET_RE = /0x([0-9A-Fa-f]+)/;

function startsWithAny(opcode: string, prefixes: string[]): boolean {
  for (const p of prefixes) {
    if (opcode.startsWith(p)) {
      return true;
    }
  }
  return false;
}

function classifyOpcode(
  op: string,
  f: SassInstructionFeatures,
  currentAddr: number,
  branchTargetAddr: number | undefined
): void {
  const opcode = op.toUpperCase();
  f.total_instructions += 1;

  // Gap 3: Hopper/Blackwell async copy (CP.ASYNC.*) — bulk and per-thread
  // global → shared pipelining.  Counted as a global load because it pulls
  // bytes out of DRAM, but tagged as async for downstream pattern analysis.
  if (opcode.startsWith("CP.ASYNC")) {
    f.global_loads += 1;
    f.async_global_loads += 1;
    if (opcode.includes(".128")) {
      f.ldg_128 += 1;
    } else if (opcode.includes(".64")) {
      f.ldg_64 += 1;
    } else if (opcode.includes(".32")) {
      f.ldg_32 += 1;
    }
  } else if (opcode.startsWith("UTMALDG")) {
    // Gap 3: TMA bulk global load (Hopper / Blackwell tensor-memory accelerator).
    // Treated as a global load so bandwidth accounting is correct on Flash-
    // Attention / CUTLASS kernels that source most of their bytes via TMA.
    f.global_loads += 1;
    f.tma_ops += 1;
  } else if (opcode.startsWith("UTMASTG")) {
    // Gap 3: TMA bulk global store.
    f.global_stores += 1;
    f.tma_ops += 1;
  } else if (opcode.startsWith("LDGSTS")) {
    // Gap 3: Ampere LDGSTS (load global → shared, async).  Treated as a
    // global load because the bytes still come from DRAM; the destination
    // (shared memory) is bookkeeping for the warp scheduler.
    f.global_loads += 1;
    f.async_global_loads += 1;
  } else if (opcode.startsWith("LDG")) {
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
    } else if (
      // Gap 2: 16-bit (half / bf16 / int16) loads.  Match BEFORE the bare ".8"
      // check below, since ".U16" contains "1" but not ".8".
      opcode.includes(".U16") ||
      opcode.includes(".S16") ||
      opcode.includes(".F16") ||
      opcode.includes(".16")
    ) {
      f.ldg_16 += 1;
    } else if (
      // Gap 2: 8-bit loads (INT8/quantised inference, char arrays).
      opcode.includes(".U8") ||
      opcode.includes(".S8") ||
      opcode.includes(".8")
    ) {
      f.ldg_8 += 1;
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
    } else if (
      opcode.includes(".U16") ||
      opcode.includes(".S16") ||
      opcode.includes(".F16") ||
      opcode.includes(".16")
    ) {
      f.stg_16 += 1;
    } else if (
      opcode.includes(".U8") ||
      opcode.includes(".S8") ||
      opcode.includes(".8")
    ) {
      f.stg_8 += 1;
    }
  } else if (opcode.startsWith("LDS")) {
    f.shared_loads += 1;
  } else if (opcode.startsWith("STS")) {
    f.shared_stores += 1;
  } else if (opcode.startsWith("LDC")) {
    // Gap 2: constant-memory loads.  LDC pulls broadcast values from the
    // per-kernel constant bank; cached at L1, not global memory.  Counted
    // separately so the global byte estimate stays honest.
    f.const_loads += 1;
  } else if (opcode.startsWith("LDL")) {
    // Local-memory load: restores a value that the compiler spilled from the
    // register file into per-thread local memory (backed by L1/L2/DRAM).
    // Even a few LDL per warp thread can stall execution for 100+ cycles.
    f.local_loads += 1;
  } else if (opcode.startsWith("STL")) {
    // Local-memory store: the spill write that precedes an LDL reload.
    f.local_stores += 1;
  }

  // Gap 1: FP64 scalar arithmetic — DFMA / DADD / DMUL / DMNMX / DSETP.
  // Counted in *both* fp64_arith_ops (so memory_model can apply 1/32-throughput
  // weighting) and arithmetic_ops (so totals stay consistent).  Without this
  // CFD/MD kernels showed compute_ops ≈ 0 → falsely "memory_bound".
  const isFp64Arith = startsWithAny(opcode, ["DFMA", "DADD", "DMUL", "DMNMX", "DSETP"]);
  if (isFp64Arith) {
    f.fp64_arith_ops += 1;
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
    ]) ||
    isFp64Arith
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

  // Gap 3: include Hopper / Blackwell barrier opcodes:
  //   BARRIER         — cluster-wide / split barrier
  //   BMOV / BSSY     — split-barrier state machine (Hopper)
  //   WARPSYNC        — warp-level sync replacement for BAR.WARP
  //   ARRIVE / WAIT   — mbarrier arrive / mbarrier wait (cooperative groups)
  // The legacy BAR / DEPBAR / MEMBAR family is preserved.
  // Note: ATOM / RED have separate counters; ARRIVE.* tokens here only fire
  // for the bare ARRIVE opcode used by mbarrier, never for ARRIVES_FOO.
  if (
    startsWithAny(opcode, [
      "BAR",
      "DEPBAR",
      "MEMBAR",
      "BARRIER",
      "BMOV",
      "BSSY",
      "WARPSYNC",
      "ARRIVE",
      "WAIT",
    ])
  ) {
    f.barrier += 1;
  }

  // Gap 7: split RET / EXIT out of the generic branch counter.  These fire at
  // most a handful of times per kernel (one per early-return) and inflated
  // branch_density / branch_per_global_mem_op on small kernels otherwise.
  if (startsWithAny(opcode, ["RET", "EXIT"])) {
    f.kernel_exit += 1;
  } else if (startsWithAny(opcode, ["BRA", "JMP", "SSY", "SYNC"])) {
    f.branch += 1;
    // Gap 8: BRA whose target address < current address is a back-edge that
    // closes a loop.  Even when the compiler fully unrolls the visible loop,
    // any remaining backward jump is a logical iteration boundary; counting
    // them lets `over_synchronized` fire on unrolled kernels.
    if (
      opcode.startsWith("BRA") &&
      branchTargetAddr !== undefined &&
      branchTargetAddr < currentAddr
    ) {
      f.back_edges += 1;
    }
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
        const addrHex = m[1]!;
        const op = m[2]!;
        const currentAddr = parseInt(addrHex, 16);
        // Gap 8: find a hex branch target on the same line — used by
        // classifyOpcode to detect back-edges (target < current).
        let targetAddr: number | undefined;
        if (op.toUpperCase().startsWith("BRA") || op.toUpperCase().startsWith("JMP")) {
          // Skip the address-comment chunk so the regex doesn't match the PC itself.
          const rest = line.slice(line.indexOf("*/") + 2);
          const tm = BRA_TARGET_RE.exec(rest);
          if (tm) {
            targetAddr = parseInt(tm[1]!, 16);
          }
        }
        classifyOpcode(op, f, currentAddr, targetAddr);
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
    ldg_16: f.ldg_16,
    ldg_8: f.ldg_8,
    stg_128: f.stg_128,
    stg_64: f.stg_64,
    stg_32: f.stg_32,
    stg_16: f.stg_16,
    stg_8: f.stg_8,
    cg_loads: f.cg_loads,
    cs_loads: f.cs_loads,
    shared_loads: f.shared_loads,
    shared_stores: f.shared_stores,
    arithmetic_ops: f.arithmetic_ops,
    integer_ops: f.integer_ops,
    tensor_ops: f.tensor_ops,
    barrier: f.barrier,
    branch: f.branch,
    kernel_exit: f.kernel_exit,
    back_edges: f.back_edges,
    const_loads: f.const_loads,
    tma_ops: f.tma_ops,
    async_global_loads: f.async_global_loads,
    local_loads: f.local_loads,
    local_stores: f.local_stores,
    wmma_ops: f.wmma_ops,
    atomic_ops: f.atomic_ops,
    global_atomic_ops: f.global_atomic_ops,
    warp_shuffle_ops: f.warp_shuffle_ops,
    warp_vote_ops: f.warp_vote_ops,
    sfu_ops: f.sfu_ops,
    fp16_arith_ops: f.fp16_arith_ops,
    fp64_arith_ops: f.fp64_arith_ops,
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
