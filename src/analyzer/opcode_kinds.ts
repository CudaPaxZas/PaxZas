/** SASS opcode coarse classifiers (matches pattern_model._is_memory_opcode / _is_compute_opcode). */

export function isMemoryOpcode(op: string): boolean {
  const upper = op.toUpperCase();
  return (
    upper.startsWith("LD") ||
    upper.startsWith("ST") ||
    upper.startsWith("LDS") ||
    upper.startsWith("STS")
  );
}

export function isComputeOpcode(op: string): boolean {
  const upper = op.toUpperCase();
  return (
    upper.startsWith("FMA") ||
    upper.startsWith("FFMA") ||
    upper.startsWith("IMAD") ||
    upper.startsWith("IADD") ||
    upper.startsWith("FADD") ||
    upper.startsWith("MUL") ||
    upper.startsWith("FMUL") ||
    upper.startsWith("HMMA") ||
    upper.startsWith("MMA")
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
