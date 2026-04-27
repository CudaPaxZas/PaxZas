/** SASS opcode coarse classifiers (matches pattern_model._is_memory_opcode / _is_compute_opcode). */

/**
 * Coarse SASS classifier for memory opcodes used by the streaming interleave
 * score.  Matches any LD-prefixed or ST-prefixed instruction (covers LDG,
 * LDS, LDL, LDC, LDSM, LDGSTS, STG, STS, STL, …) plus the modern bulk/async
 * copy variants the LD/ST prefixes miss:
 *   - CP.ASYNC  — Hopper async global to shared copy
 *   - UTMALDG / UTMASTG — TMA bulk transfers (Gap 3)
 */
export function isMemoryOpcode(op: string): boolean {
  const upper = op.toUpperCase();
  return (
    upper.startsWith("LD") ||
    upper.startsWith("ST") ||
    upper.startsWith("CP.ASYNC") ||
    upper.startsWith("UTMALDG") ||
    upper.startsWith("UTMASTG")
  );
}

/**
 * Coarse SASS classifier for "compute" instructions used by the streaming
 * interleave score and Group E stall heuristics.
 *
 * Gap 4: the original list (`FMA / FFMA / IMAD / IADD / FADD / MUL / FMUL /
 * HMMA / MMA`) missed every modern non-FP32 compute instruction:
 *   - FP16 scalar:  HFMA / HADD / HMUL
 *   - FP64 scalar:  DFMA / DADD / DMUL
 *   - Integer ALU:  IMUL / IMNMX / ISCADD / ISET / ICMP / IXOR / IAND / IOR /
 *                   ISHL / ISHR
 *   - Tensor cores: WGMMA (Hopper warp-group MMA), IMMA (int tensor),
 *                   BMMA (binary tensor)
 *   - SFU:          MUFU (transcendentals — debatable but really *is* compute)
 *
 * Effect of the missing entries: in HFMA/IMMA-heavy kernels the compute side
 * was invisible, the interleave score stayed near 0 ("stacked"), and the E1
 * stall heuristic mis-fired because every load looked like an isolated
 * latency chain.
 */
export function isComputeOpcode(op: string): boolean {
  const upper = op.toUpperCase();
  return (
    upper.startsWith("FFMA") ||
    upper.startsWith("FMA") ||
    upper.startsWith("FADD") ||
    upper.startsWith("FMUL") ||
    // FP16 scalar
    upper.startsWith("HFMA") ||
    upper.startsWith("HADD") ||
    upper.startsWith("HMUL") ||
    // FP64 scalar
    upper.startsWith("DFMA") ||
    upper.startsWith("DADD") ||
    upper.startsWith("DMUL") ||
    // Integer ALU / address arithmetic
    upper.startsWith("IADD") ||
    upper.startsWith("IMAD") ||
    upper.startsWith("IMUL") ||
    upper.startsWith("IMNMX") ||
    upper.startsWith("ISCADD") ||
    upper.startsWith("ISETP") ||
    upper.startsWith("ISET") ||
    upper.startsWith("ICMP") ||
    upper.startsWith("IXOR") ||
    upper.startsWith("IAND") ||
    upper.startsWith("IOR") ||
    upper.startsWith("ISHL") ||
    upper.startsWith("ISHR") ||
    // Bare MUL fallback (rare — mostly old SASS dumps)
    upper.startsWith("MUL") ||
    // Tensor cores (FP, INT, binary)
    upper.startsWith("WGMMA") ||
    upper.startsWith("HMMA") ||
    upper.startsWith("IMMA") ||
    upper.startsWith("BMMA") ||
    upper.startsWith("MMA") ||
    // Special function unit (transcendentals)
    upper.startsWith("MUFU")
  );
}

export function safeDiv(n: number, d: number): number {
  if (d === 0) {
    return 0;
  }
  return n / d;
}

export interface InterleaveStreamState {
  prevKind: string | undefined;
  transitions: number;
  opportunities: number;
  maxLoadRun: number;
  loadRun: number;
}

export function createInterleaveState(): InterleaveStreamState {
  return {
    prevKind: undefined,
    transitions: 0,
    opportunities: 0,
    maxLoadRun: 0,
    loadRun: 0,
  };
}

/** One SASS opcode observed: updates interleave + consecutive-LD streak (pattern_model semantics). */
export function observeSassOpcodeForPatternMetrics(
  op: string,
  s: InterleaveStreamState
): void {
  const upper = op.toUpperCase();
  let kind: string | undefined;
  if (isMemoryOpcode(op)) {
    kind = "mem";
  } else if (isComputeOpcode(op)) {
    kind = "compute";
  }
  if (kind !== undefined) {
    if (s.prevKind !== undefined) {
      s.opportunities += 1;
      if (kind !== s.prevKind) {
        s.transitions += 1;
      }
    }
    s.prevKind = kind;
  }
  if (upper.startsWith("LD")) {
    s.loadRun += 1;
    if (s.loadRun > s.maxLoadRun) {
      s.maxLoadRun = s.loadRun;
    }
  } else {
    s.loadRun = 0;
  }
}

export function interleaveScoreFromState(s: InterleaveStreamState): number {
  return safeDiv(s.transitions, s.opportunities);
}
