/**
 * Tests for the coarse SASS classifiers in src/analyzer/opcode_kinds.ts.
 *
 * These classifiers (`isMemoryOpcode`, `isComputeOpcode`) feed the streaming
 * interleave score and the Group E stall heuristics in pattern_model. Bugs
 * here propagate into stall_memory_dependency, stall_memory_throttle, and
 * the interleaving classification (interleaved/stacked/mixed).
 *
 * Gap 4 expanded `isComputeOpcode` to recognise modern non-FP32 compute work
 * (FP16 / FP64 scalar, integer ALU, modern tensor cores, MUFU); these tests
 * lock in that behaviour.
 *
 * Gap 3 cleanup expanded `isMemoryOpcode` to recognise modern bulk/async
 * memory operations (CP.ASYNC, UTMALDG, UTMASTG) that don't use the LD/ST
 * prefix; covered here too.
 */
import { describe, expect, it } from "vitest";
import {
  createInterleaveState,
  interleaveScoreFromState,
  isComputeOpcode,
  isMemoryOpcode,
  observeSassOpcodeForPatternMetrics,
} from "../src/analyzer/opcode_kinds";

describe("opcode_kinds — isMemoryOpcode (Gap 3 modern memory ops)", () => {
  it("classifies LD-prefixed opcodes as memory", () => {
    expect(isMemoryOpcode("LDG.E.32")).toBe(true);
    expect(isMemoryOpcode("LDS.128")).toBe(true);
    expect(isMemoryOpcode("LDL")).toBe(true);
    expect(isMemoryOpcode("LDC")).toBe(true);
    expect(isMemoryOpcode("LDSM")).toBe(true);
    expect(isMemoryOpcode("LDGSTS.E.128")).toBe(true);
  });

  it("classifies ST-prefixed opcodes as memory", () => {
    expect(isMemoryOpcode("STG.E.SYS")).toBe(true);
    expect(isMemoryOpcode("STS.32")).toBe(true);
    expect(isMemoryOpcode("STL")).toBe(true);
  });

  it("classifies CP.ASYNC and TMA bulk copies as memory (Gap 3)", () => {
    // CP.ASYNC isn't LD/ST-prefixed; UTMALDG/UTMASTG aren't either.
    expect(isMemoryOpcode("CP.ASYNC.CG.16")).toBe(true);
    expect(isMemoryOpcode("CP.ASYNC.CA.4")).toBe(true);
    expect(isMemoryOpcode("UTMALDG.2D")).toBe(true);
    expect(isMemoryOpcode("UTMASTG.2D")).toBe(true);
  });

  it("does not classify arithmetic opcodes as memory", () => {
    expect(isMemoryOpcode("FFMA")).toBe(false);
    expect(isMemoryOpcode("DFMA")).toBe(false);
    expect(isMemoryOpcode("HFMA")).toBe(false);
    expect(isMemoryOpcode("HMMA.16816.F32")).toBe(false);
    expect(isMemoryOpcode("MUFU.SIN")).toBe(false);
    // BAR, BRA, RET — control-flow / barrier, not memory.
    expect(isMemoryOpcode("BAR.SYNC")).toBe(false);
    expect(isMemoryOpcode("BRA")).toBe(false);
    expect(isMemoryOpcode("RET")).toBe(false);
  });
});

describe("opcode_kinds — isComputeOpcode (Gap 4 expansion)", () => {
  it("recognises FP32 scalar arithmetic", () => {
    expect(isComputeOpcode("FFMA")).toBe(true);
    expect(isComputeOpcode("FFMA.RN")).toBe(true);
    expect(isComputeOpcode("FADD")).toBe(true);
    expect(isComputeOpcode("FMUL")).toBe(true);
    expect(isComputeOpcode("FMA.F32")).toBe(true);
  });

  it("recognises FP16 scalar arithmetic (Gap 4)", () => {
    // These were silently dropped before Gap 4, hiding compute work in HFMA-heavy
    // kernels and pushing the streaming interleave score toward "stacked".
    expect(isComputeOpcode("HFMA")).toBe(true);
    expect(isComputeOpcode("HADD")).toBe(true);
    expect(isComputeOpcode("HMUL")).toBe(true);
  });

  it("recognises FP64 scalar arithmetic (Gap 4)", () => {
    expect(isComputeOpcode("DFMA")).toBe(true);
    expect(isComputeOpcode("DADD")).toBe(true);
    expect(isComputeOpcode("DMUL")).toBe(true);
  });

  it("recognises integer ALU instructions (Gap 4)", () => {
    expect(isComputeOpcode("IADD")).toBe(true);
    expect(isComputeOpcode("IMAD")).toBe(true);
    expect(isComputeOpcode("IMUL")).toBe(true);
    expect(isComputeOpcode("IMNMX")).toBe(true);
    expect(isComputeOpcode("ISCADD")).toBe(true);
    expect(isComputeOpcode("ISETP.LT.AND")).toBe(true);
    expect(isComputeOpcode("ISET.GE")).toBe(true);
    expect(isComputeOpcode("ICMP")).toBe(true);
    expect(isComputeOpcode("IXOR")).toBe(true);
    expect(isComputeOpcode("IAND")).toBe(true);
    expect(isComputeOpcode("IOR")).toBe(true);
    expect(isComputeOpcode("ISHL")).toBe(true);
    expect(isComputeOpcode("ISHR")).toBe(true);
  });

  it("recognises modern tensor-core ops (Gap 4)", () => {
    expect(isComputeOpcode("HMMA.16816.F32")).toBe(true);
    expect(isComputeOpcode("MMA.884.F16")).toBe(true);
    expect(isComputeOpcode("WGMMA.64816.F32.F16.F16")).toBe(true);
    expect(isComputeOpcode("IMMA.16816.S8")).toBe(true);
    expect(isComputeOpcode("BMMA.88128.B1")).toBe(true);
  });

  it("recognises MUFU as compute (transcendentals are real ALU work)", () => {
    expect(isComputeOpcode("MUFU.SIN")).toBe(true);
    expect(isComputeOpcode("MUFU.RCP")).toBe(true);
    expect(isComputeOpcode("MUFU.SQRT")).toBe(true);
  });

  it("does not classify memory or control-flow opcodes as compute", () => {
    expect(isComputeOpcode("LDG.E.32")).toBe(false);
    expect(isComputeOpcode("STG.E.SYS")).toBe(false);
    expect(isComputeOpcode("LDS.128")).toBe(false);
    expect(isComputeOpcode("STS.32")).toBe(false);
    expect(isComputeOpcode("CP.ASYNC.CG.16")).toBe(false);
    expect(isComputeOpcode("BAR.SYNC")).toBe(false);
    expect(isComputeOpcode("BRA")).toBe(false);
    expect(isComputeOpcode("JMP")).toBe(false);
    expect(isComputeOpcode("RET")).toBe(false);
    expect(isComputeOpcode("EXIT")).toBe(false);
    expect(isComputeOpcode("SHFL.SYNC.IDX")).toBe(false);
    expect(isComputeOpcode("VOTE.SYNC.ALL")).toBe(false);
    expect(isComputeOpcode("ATOM.E.ADD")).toBe(false);
  });
});

describe("opcode_kinds — Gap 4 effect on stream_interleave_score", () => {
  // Before Gap 4, HFMA was not recognised as compute, so an LDG/HFMA stream
  // looked like "all memory" and produced an interleave score of 0 (stacked).
  // After Gap 4, the same stream alternates mem/compute and scores 1.0.
  it("HFMA / LDG alternation produces high interleave score", () => {
    const s = createInterleaveState();
    const stream = [
      "LDG.E.32",
      "HFMA",
      "LDG.E.32",
      "HFMA",
      "LDG.E.32",
      "HFMA",
    ];
    for (const op of stream) {
      observeSassOpcodeForPatternMetrics(op, s);
    }
    // 6 opcodes → 5 transitions evaluated, all alternate → score 1.0
    expect(interleaveScoreFromState(s)).toBeCloseTo(1.0, 5);
  });

  it("DFMA / LDG alternation produces high interleave score (FP64 case)", () => {
    const s = createInterleaveState();
    for (const op of ["LDG.E.32", "DFMA", "LDG.E.32", "DFMA", "LDG.E.32", "DFMA"]) {
      observeSassOpcodeForPatternMetrics(op, s);
    }
    expect(interleaveScoreFromState(s)).toBeCloseTo(1.0, 5);
  });

  it("HMMA / LDG alternation produces high interleave score (tensor case)", () => {
    const s = createInterleaveState();
    for (const op of [
      "LDG.E.128",
      "HMMA.16816.F32",
      "LDG.E.128",
      "HMMA.16816.F32",
      "LDG.E.128",
      "HMMA.16816.F32",
    ]) {
      observeSassOpcodeForPatternMetrics(op, s);
    }
    expect(interleaveScoreFromState(s)).toBeCloseTo(1.0, 5);
  });

  it("pure LDG stream still scores 0 (sanity — no compute interleaved)", () => {
    const s = createInterleaveState();
    for (let i = 0; i < 8; i++) {
      observeSassOpcodeForPatternMetrics("LDG.E.32", s);
    }
    // All 8 ops are memory; no transitions → score 0.
    expect(interleaveScoreFromState(s)).toBe(0);
    // And maxLoadRun should track the consecutive-load streak for E1 stall.
    expect(s.maxLoadRun).toBe(8);
  });
});
