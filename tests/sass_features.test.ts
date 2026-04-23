/**
 * Mirrors tests/test_sass_features.py when fixture file is absent:
 * uses minimal inline SASS with expected opcode classes.
 */
import { describe, expect, it } from "vitest";
import {
  detectSassSmTargets,
  extractSassFeatures,
  inferRegistersPerThreadFromSass,
} from "../src/analyzer/sass_features";
import { sassHx } from "./helpers/ir_fixtures";

/** Minimal dump shaped like cuobjdump (matches _SASS_OPCODE_RE). */
function inlineSassFixture(kernel: string): string {
  const lines: string[] = [`Function : ${kernel}`, ""];
  let a = 0x0100;
  lines.push(`${sassHx(a)} STG.E.SYS [R2], R4;`);
  a += 0x10;
  lines.push(`${sassHx(a)} IADD R0, R1, R2;`);
  a += 0x10;
  lines.push(`${sassHx(a)} BRA 0x200;`);
  return lines.join("\n");
}

const KERNEL = "_ZN4test7kernels6sampleEv";

describe("sass_features (Python parity shape)", () => {
  it("extract_sass_features_counts_basic_op_classes", () => {
    const text = inlineSassFixture(KERNEL);
    const [name, f] = extractSassFeatures(text, KERNEL);
    expect(name).toBe(KERNEL);
    expect(f.global_loads).toBe(0);
    expect(f.global_stores).toBe(1);
    expect(f.shared_loads).toBe(0);
    expect(f.shared_stores).toBe(0);
    expect(f.arithmetic_ops).toBeGreaterThanOrEqual(1);
    expect(f.tensor_ops).toBe(0);
    expect(f.barrier).toBe(0);
    expect(f.branch).toBeGreaterThanOrEqual(1);
  });

  it("infer_registers_per_thread_from_sass", () => {
    const text = inlineSassFixture(KERNEL);
    const regs = inferRegistersPerThreadFromSass(text, KERNEL);
    expect(regs).toBeDefined();
    expect(regs!).toBeGreaterThanOrEqual(1);
  });

  it("detects multiple architecture targets in fatbin-style SASS text", () => {
    const text = [
      "Fatbin elf code:",
      "code for sm_80",
      "Function : _Ztest80",
      "code for sm_90",
      "Function : _Ztest90",
      "arch = sm_90",
      "arch = sm_120",
    ].join("\n");
    expect(detectSassSmTargets(text)).toEqual([80, 90, 120]);
  });
});

describe("sass_features — predicated branches", () => {
  it("predicated_bra_counted_as_branch", () => {
    // @P0 BRA and @!P1 BRA are warp-divergence instructions.
    // Without the predicate-strip fix in SASS_OPCODE_RE they were silently
    // dropped and branch would be 0.
    const lines = [
      "Function : _ZN4pred7kernelEv",
      "",
      `    /*0000*/ LDG.E.32 R0, [R4];`,
      `    /*0010*/ @P0 BRA 0x100;`,    // predicated branch (taken if P0 is true)
      `    /*0020*/ @!P1 BRA 0x200;`,   // negated-predicate branch
      `    /*0030*/ FFMA.RN R2, R0, R1, R2;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.branch).toBe(2);            // both predicated branches counted
    expect(f.global_loads).toBe(1);      // LDG still counted normally
    expect(f.arithmetic_ops).toBe(1);    // FFMA still counted normally
  });

  it("non_predicated_bra_still_counted", () => {
    // Regression: plain (non-predicated) BRA must still work after regex change.
    const lines = [
      "Function : _ZN4plain7kernelEv",
      "",
      `    /*0000*/ BRA 0x300;`,
      `    /*0010*/ @P0 BRA 0x400;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.branch).toBe(2);
  });
});

describe("sass_features — register spill detection (LDL/STL)", () => {
  it("ldl_stl_counted_as_local_memory", () => {
    // LDL/STL prove the compiler spilled registers to local memory.
    // They must NOT be counted as global_loads/global_stores.
    const lines = [
      "Function : _ZN4spill7kernelEv",
      "",
      `    /*0000*/ LDL R0, [R2+0x10];`,   // local load  (spill restore)
      `    /*0010*/ STL [R4+0x20], R6;`,   // local store (spill save)
      `    /*0020*/ LDL R8, [R10];`,       // second local load
      `    /*0030*/ FFMA.RN R2, R0, R1, R2;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.local_loads).toBe(2);
    expect(f.local_stores).toBe(1);
    // Must not bleed into global memory counters
    expect(f.global_loads).toBe(0);
    expect(f.global_stores).toBe(0);
  });

  it("no_spill_when_no_ldl_stl", () => {
    // Sanity check: without LDL/STL the spill counters are zero.
    const lines = [
      "Function : _ZN4clean7kernelEv",
      "",
      `    /*0000*/ LDG.E.32 R0, [R4];`,
      `    /*0010*/ STG.E.32 [R2], R6;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.local_loads).toBe(0);
    expect(f.local_stores).toBe(0);
  });
});

describe("sass_features — tensor core detection (HMMA/WGMMA/WMMA)", () => {
  it("hmma_counts_as_tensor_ops_and_wmma_ops", () => {
    // HMMA is the Volta/Ampere hardware encoding of WMMA fragment operations.
    // It must be counted in tensor_ops AND wmma_ops (FP matrix units).
    const lines = [
      "Function : _ZN6tensor7kernelEv",
      "",
      `    /*0000*/ HMMA.16816.F32 {R0,R1,R2,R3},{R4,R5},{R6,R7},{R0,R1,R2,R3};`,
      `    /*0010*/ HMMA.16816.F32 {R8,R9,R10,R11},{R4,R5},{R6,R7},{R8,R9,R10,R11};`,
      // IMMA = integer tensor op; must count in tensor_ops but NOT wmma_ops
      `    /*0020*/ IMMA.16816.S8 {R12,R13},{R4,R5},{R6,R7},{R12,R13};`,
      `    /*0030*/ FFMA.RN R0, R1, R2, R3;`,   // scalar FFMA — not a tensor op
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.tensor_ops).toBe(3);   // HMMA x2 + IMMA x1
    expect(f.wmma_ops).toBe(2);     // only the two HMMA
    expect(f.arithmetic_ops).toBe(1); // FFMA counts as arithmetic, not tensor
  });

  it("wgmma_counts_as_wmma_ops", () => {
    // WGMMA is the Hopper (SM90) warp-group MMA instruction.
    const lines = [
      "Function : _ZN7hopper7kernelEv",
      "",
      `    /*0000*/ WGMMA.64816.F32.F16.F16;`,
      `    /*0010*/ LDG.E.32 R0, [R4];`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.tensor_ops).toBe(1);
    expect(f.wmma_ops).toBe(1);
  });

  it("mma_counts_as_tensor_ops_not_wmma_ops", () => {
    // Plain MMA (Ampere sm_80 integer path) counts in tensor_ops but not wmma_ops.
    const lines = [
      "Function : _ZN3mma7kernelEv",
      "",
      `    /*0000*/ MMA.884.F16.F16;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.tensor_ops).toBe(1);
    expect(f.wmma_ops).toBe(0);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// New counter tests: atomic_ops / sfu_ops / fp16_arith_ops
// ─────────────────────────────────────────────────────────────────────────────

describe("sass_features — atomic/SFU/FP16-scalar detection", () => {
  // ── atomic_ops ────────────────────────────────────────────────────────────

  it("atom_and_red_counted_as_atomic_ops", () => {
    // ATOM.* = global atomic read-modify-write (ADD, CAS, MAX, …) — serialised
    // at the L2 cache.  RED.* = fire-and-forget reduction (no return value) but
    // still serialised at L2.  Both should be tallied in atomic_ops AND in
    // global_atomic_ops; a plain global load (LDG) is not an atomic.
    const lines = [
      "Function : _ZN4atom7kernelEv",
      "",
      `    /*0000*/ ATOM.E.ADD [R2], R4;`,     // global add; returns result
      `    /*0010*/ ATOM.E.CAS [R6], R8, R2;`, // compare-and-swap
      `    /*0020*/ RED.E.ADD [R10], R12;`,     // fire-and-forget reduction
      `    /*0030*/ LDG.E.32 R0, [R4];`,        // plain load — not atomic
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.atomic_ops).toBe(3);        // 2 ATOM + 1 RED
    expect(f.global_atomic_ops).toBe(3); // same 3 — all are global
    expect(f.global_loads).toBe(1);      // the LDG only; atomics must not bleed in
  });

  it("atoms_shared_memory_counted_in_total_not_global", () => {
    // ATOMS = shared-memory atomic; causes shared-bank serialisation only,
    // NOT L2 contention.  It must appear in atomic_ops (total serialisation
    // pressure) but NOT in global_atomic_ops (L2 contention numerator).
    const lines = [
      "Function : _ZN5atoms7kernelEv",
      "",
      `    /*0000*/ ATOMS.ADD [R2], R4;`,
      `    /*0010*/ ATOMS.ADD [R6], R8;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.atomic_ops).toBe(2);        // total: both ATOMS counted
    expect(f.global_atomic_ops).toBe(0); // L2 global: none — ATOMS is shared
  });

  it("no_atomic_ops_for_plain_loads_stores", () => {
    // LDG/STG are non-atomic global accesses; FFMA is pure arithmetic.
    // None of these should increment atomic_ops or global_atomic_ops.
    const lines = [
      "Function : _ZN6noatom7kernelEv",
      "",
      `    /*0000*/ LDG.E.32 R0, [R4];`,
      `    /*0010*/ STG.E.SYS [R2], R4;`,
      `    /*0020*/ FFMA.FTZ R0, R1, R2, R3;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.atomic_ops).toBe(0);
    expect(f.global_atomic_ops).toBe(0);
  });

  // ── sfu_ops ───────────────────────────────────────────────────────────────

  it("mufu_variants_counted_as_sfu_ops", () => {
    // MUFU = Multi-Function Unit instruction; handles all hardware transcendentals
    // (SIN, COS, EXP2, LOG2, RCP, RSQ, SQRT).  Each MUFU dispatches to a
    // dedicated pipeline that has 1/4 the throughput of the FP32 ALU on Ampere.
    // An FFMA in the same kernel is pure ALU and must NOT count as an SFU op.
    const lines = [
      "Function : _ZN3sfu7kernelEv",
      "",
      `    /*0000*/ MUFU.SIN R0, R1;`,
      `    /*0010*/ MUFU.RCP R2, R3;`,
      `    /*0020*/ MUFU.SQRT R4, R5;`,
      `    /*0030*/ MUFU.EXP2 R6, R7;`,
      `    /*0040*/ FFMA.FTZ R8, R0, R2, R4;`, // ALU — not an SFU op
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.sfu_ops).toBe(4);
    // The FFMA is the only arithmetic op; MUFU instructions are separated
    // into their own counter and do NOT double-count into arithmetic_ops.
    expect(f.arithmetic_ops).toBe(1);
  });

  it("no_sfu_ops_without_mufu", () => {
    // Kernels with only FP32 arithmetic should report zero SFU ops.
    const lines = [
      "Function : _ZN5nosfu7kernelEv",
      "",
      `    /*0000*/ FFMA.FTZ R0, R1, R2, R3;`,
      `    /*0010*/ LDG.E.32 R4, [R2];`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.sfu_ops).toBe(0);
  });

  // ── fp16_arith_ops ────────────────────────────────────────────────────────

  it("hfma_hadd_hmul_counted_as_fp16_arith_ops", () => {
    // HFMA/HADD/HMUL are scalar FP16 CUDA-core instructions (NOT tensor-core).
    // They increment fp16_arith_ops to separate them from FP32 work.
    // Because they are still arithmetic, they ALSO increment arithmetic_ops —
    // fp16_arith_ops is a strict subset of arithmetic_ops.
    const lines = [
      "Function : _ZN4fp16kernelEv",
      "",
      `    /*0000*/ HFMA.F16 R0, R1, R2, R3;`,
      `    /*0010*/ HADD.F16 R4, R5, R6;`,
      `    /*0020*/ HMUL.F16 R7, R8, R9;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.fp16_arith_ops).toBe(3);
    // fp16 is a subset: arithmetic_ops must equal fp16_arith_ops when no FP32 present
    expect(f.arithmetic_ops).toBe(3);
  });

  it("fp16_is_strict_subset_of_arithmetic_ops", () => {
    // Mix of FP32 (FFMA) and FP16 (HFMA).  arithmetic_ops counts both;
    // fp16_arith_ops counts only the HFMA.
    const lines = [
      "Function : _ZN5mixedkernelEv",
      "",
      `    /*0000*/ FFMA.FTZ R0, R1, R2, R3;`,
      `    /*0010*/ FFMA.FTZ R4, R5, R6, R7;`,
      `    /*0020*/ HFMA.F16 R8, R9, R10, R11;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.arithmetic_ops).toBe(3);  // 2 FFMA + 1 HFMA
    expect(f.fp16_arith_ops).toBe(1);  // only the one HFMA
  });

  it("ffma_does_not_increment_fp16_arith_ops", () => {
    // Canonical FP32 operations must never bleed into fp16_arith_ops.
    const lines = [
      "Function : _ZN4ffmakernelEv",
      "",
      `    /*0000*/ FFMA.FTZ R0, R1, R2, R3;`,
      `    /*0010*/ FFMA.FTZ R4, R5, R6, R7;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.fp16_arith_ops).toBe(0);
    expect(f.arithmetic_ops).toBe(2); // both FFMAs still counted
  });

  // ── warp_shuffle_ops ──────────────────────────────────────────────────────

  it("shfl_instructions_counted_as_warp_shuffle_ops", () => {
    // SHFL.SYNC.IDX and SHFL.SYNC.UP are both warp-shuffle instructions.
    const lines = [
      "Function : _ZN5wshflkernelEv",
      "",
      `    /*0000*/ SHFL.SYNC.IDX R0, R1, R2, 0x1f;`,
      `    /*0010*/ SHFL.SYNC.UP  R3, R4, R5, 0x0;`,
      `    /*0020*/ FFMA.FTZ R6, R7, R8, R9;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.warp_shuffle_ops).toBe(2);
  });

  it("non_shfl_instructions_do_not_increment_warp_shuffle_ops", () => {
    const lines = [
      "Function : _ZN6noshflkernelEv",
      "",
      `    /*0000*/ FFMA.FTZ R0, R1, R2, R3;`,
      `    /*0010*/ LDG.E.32 R4, [R2];`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.warp_shuffle_ops).toBe(0);
  });

  // ── warp_vote_ops ─────────────────────────────────────────────────────────

  it("vote_and_match_instructions_counted_as_warp_vote_ops", () => {
    // VOTE.ALL/ANY/EQ and MATCH.ANY/ALL are warp-vote primitives.
    const lines = [
      "Function : _ZN5wvotekernelEv",
      "",
      `    /*0000*/ VOTE.SYNC.ALL P0, P1, 0xff;`,
      `    /*0010*/ MATCH.ANY.SYNC R0, R1, 0xff;`,
      `    /*0020*/ VOTE.SYNC.ANY P2, P3, 0xff;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.warp_vote_ops).toBe(3);
  });

  it("non_vote_instructions_do_not_increment_warp_vote_ops", () => {
    const lines = [
      "Function : _ZN6novotekernelEv",
      "",
      `    /*0000*/ FFMA.FTZ R0, R1, R2, R3;`,
      `    /*0010*/ BAR.SYNC 0;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.warp_vote_ops).toBe(0);
  });

  it("integer_opcodes_counted_as_integer_ops", () => {
    const lines = [
      "Function : _ZN7intops7kernelEv",
      "",
      `    /*0000*/ IADD R0, R1, R2;`,
      `    /*0010*/ IMAD R3, R4, R5, R6;`,
      `    /*0020*/ ISET.LT.AND P0, PT, R7, R8, PT;`,
      `    /*0030*/ IXOR R9, R10, R11;`,
      `    /*0040*/ ISHL R12, R13, 0x2;`,
      `    /*0050*/ FFMA.FTZ R0, R1, R2, R3;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.integer_ops).toBe(5);
    expect(f.arithmetic_ops).toBe(3);
  });

  it("fp_only_kernel_has_zero_integer_ops", () => {
    const lines = [
      "Function : _ZN6fponly7kernelEv",
      "",
      `    /*0000*/ FFMA.FTZ R0, R1, R2, R3;`,
      `    /*0010*/ FADD R4, R5, R6;`,
      `    /*0020*/ FMUL R7, R8, R9;`,
    ].join("\n");
    const [, f] = extractSassFeatures(lines, undefined);
    expect(f.integer_ops).toBe(0);
    expect(f.arithmetic_ops).toBe(3);
  });
});
