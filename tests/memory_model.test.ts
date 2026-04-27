/**
 * Mirrors tests/test_memory_model.py
 */
import { describe, expect, it } from "vitest";
import { analyzeMemory } from "../src/analyzer/memory_model";
import {
  PTX_HEAD,
  featuresFromPtx,
  featuresFromSass,
  sassHx,
} from "./helpers/ir_fixtures";

describe("memory_model (Python parity)", () => {
  it("classifies_memory_bound_low_intensity", () => {
    const loads = Array.from({ length: 16 }, () => "ld.global.f32 %f1, [%rd1];").join(
      "\n  "
    );
    const stores = Array.from({ length: 8 }, () => "st.global.f32 [%rd2], %f1;").join(
      "\n  "
    );
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z7m_bound(
  .param .u64 p
)
{
  .reg .f32 %f<4>;
  .reg .u64 %rd<4>;
  ${loads}
  ${stores}
  add.f32 %f2, %f1, %f1;
  add.f32 %f3, %f2, %f2;
  mul.f32 %f4, %f3, %f3;
  ret;
}
`;
    const ptx = featuresFromPtx(ptxMod, "_Z7m_bound");
    const out = analyzeMemory(ptx);
    expect(["memory_bound", "balanced"]).toContain(out.class);
    expect(out.global_mem_ops).toBe(24);
    expect(out.global_mem_source).toBe("ptx");
    expect(out.compute_source).toBe("ptx");
    expect(out.shared_mem_source).toBe("unknown");
    expect(out).toHaveProperty("reuse_strength");
    expect(out).toHaveProperty("mem_compute_ratio");
    expect(out.memory_pressure).toBe(out.bytes_proxy);
  });

  it("uses SASS shared ops for reuse_ratio", () => {
    const adds = Array.from({ length: 20 }, () => "add.f32 %f1, %f2, %f3;").join("\n  ");
    const muls = Array.from({ length: 20 }, () => "mul.f32 %f4, %f1, %f2;").join("\n  ");
    const fmas = Array.from(
      { length: 10 },
      () => "fma.rn.f32 %f5, %f1, %f2, %f3;"
    ).join("\n  ");
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z6reuse(
  .param .u64 p
)
{
  .reg .f32 %f<8>;
  .reg .u64 %rd<4>;
  ld.global.f32 %f1, [%rd1];
  ld.global.f32 %f2, [%rd1];
  st.global.f32 [%rd2], %f1;
  st.global.f32 [%rd2], %f2;
  ${adds}
  ${muls}
  ${fmas}
  ret;
}
`;
    const ptx = featuresFromPtx(ptxMod, "_Z6reuse");

    let a = 0x1000;
    const lines: string[] = ["Function : _Z6reuse", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
    a += 0x10;
    lines.push(`${sassHx(a)} STG.E.32 [R2], R4;`);
    a += 0x10;
    for (let i = 0; i < 30; i++) {
      lines.push(`${sassHx(a)} LDS.128 R8, [R4];`);
      a += 0x10;
    }
    for (let i = 0; i < 10; i++) {
      lines.push(`${sassHx(a)} STS.128 [R4], R8;`);
      a += 0x10;
    }
    for (let i = 0; i < 5; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const sass = featuresFromSass(lines.join("\n"), "_Z6reuse");

    const out = analyzeMemory(ptx, sass);
    expect(out.shared_mem_ops).toBe(40);
    expect(out.reuse_ratio).toBeGreaterThan(2.0);
    expect(out.reuse_strength).toBeGreaterThan(1.0);
    expect(out.global_mem_source).toBe("sass");
    expect(out.shared_mem_source).toBe("sass");
    expect(out.compute_source).toBe("sass");
    expect(out.cache_policy).toBeNull();
    expect(out.confidence).toBeGreaterThanOrEqual(0.8);
    expect(out.class).toBe("reuse_optimized");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// New tests covering Gaps 1-6
// ─────────────────────────────────────────────────────────────────────────────

describe("memory_model — gap fixes", () => {
  // ── Gap 1: Tensor FLOP proxy ────────────────────────────────────────────────
  //
  // A single HMMA.16816 instruction performs a 16×16×16 matrix MAC ≈ 512 FLOPs.
  // The old formula used `2 × tensor_ops` (≈ 256× undercount), causing tensor-
  // heavy kernels to appear near-zero intensity → incorrectly "memory_bound".

  it("tensor_heavy_kernel_has_high_arithmetic_intensity", () => {
    // 4 LDG.128 loads + 16 HMMA tensor-core instructions.
    // Old: sassFlopsProxy = 0 + 2×16 = 32; bytes = 4×16 = 64; intensity = 0.5
    // New: sassFlopsProxy = 0 + 16×512 = 8192; bytes = 64; intensity = 128 >> 2.0
    // Should classify as compute_friendly, NOT memory_bound.
    let a = 0x3000;
    const lines = ["Function : _Z6tensorK", ""];
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} LDG.E.128 R${i * 4}, [R${i * 4 + 16}];`);
      a += 0x10;
    }
    for (let i = 0; i < 16; i++) {
      lines.push(
        `${sassHx(a)} HMMA.16816.F32 {R0,R1,R2,R3},{R4,R5},{R6,R7},{R0,R1,R2,R3};`
      );
      a += 0x10;
    }
    const ptxMod =
      PTX_HEAD +
      `\n.visible .entry _Z6tensorK(.param .u64 p) { .reg .f32 %f<8>; ret; }\n`;
    const ptx  = featuresFromPtx(ptxMod, "_Z6tensorK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6tensorK");
    const out  = analyzeMemory(ptx, sass);
    // With correct FLOP proxy intensity >> 2.0 → compute_friendly
    expect(out.class).toBe("compute_friendly");
    // Sanity: intensity must be well above the memory-bound threshold
    const intensity = out.arithmetic_intensity_ops_per_byte;
    expect(intensity).not.toBeNull();
    expect(intensity!).toBeGreaterThan(10);
  });

  // ── Gap 2: SFU ops in FLOP estimate ────────────────────────────────────────
  //
  // MUFU.* transcendental instructions are real arithmetic work.  Excluding
  // them from the FLOP estimate makes SFU-heavy kernels (sigmoid, GELU, …)
  // appear nearly compute-free → falsely memory_bound.

  it("sfu_heavy_kernel_not_misclassified_memory_bound", () => {
    // 2 global loads + 20 MUFU instructions (heavy SFU work).
    // Old: sassFlopsProxy = 0 + 0 = 0 → intensity = 0 → memory_bound
    // New: sassFlopsProxy = 20×4 = 80; bytes = 2×4 = 8; intensity = 10 → compute_friendly
    let a = 0x3100;
    const lines = ["Function : _Z5sfuHvK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    lines.push(`${sassHx(a)} LDG.E.32 R1, [R3];`);  a += 0x10;
    for (let i = 0; i < 20; i++) {
      lines.push(`${sassHx(a)} MUFU.SIN R${i % 8}, R${(i % 8) + 1};`);
      a += 0x10;
    }
    const ptxMod =
      PTX_HEAD +
      `\n.visible .entry _Z5sfuHvK(.param .u64 p) { .reg .f32 %f<4>; ret; }\n`;
    const ptx  = featuresFromPtx(ptxMod, "_Z5sfuHvK");
    const sass = featuresFromSass(lines.join("\n"), "_Z5sfuHvK");
    const out  = analyzeMemory(ptx, sass);
    expect(out.class).not.toBe("memory_bound");
    const intensity = out.arithmetic_intensity_ops_per_byte;
    expect(intensity).not.toBeNull();
    expect(intensity!).toBeGreaterThan(2);
  });

  // ── Gap 3: Dynamic pressure threshold (globalMemOps >= 4) ──────────────────
  //
  // The old fixed threshold (2048 bytes) excluded small-but-clearly-memory-bound
  // kernels.  A kernel with 24 global ops × 4 bytes = 96 bytes moved and
  // intensity ≈ 0.03 should be "memory_bound", not "balanced".

  it("small_low_intensity_kernel_classified_memory_bound", () => {
    // 16 loads + 8 stores = 24 global ops, 96 bytes, 3 arithmetic ops.
    // intensity = 3/96 ≈ 0.03 << 0.5; globalMemOps = 24 >= 4 → memory_bound.
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z7smMemBK(.param .u64 p) {
  .reg .f32 %f<4>; .reg .u64 %rd<4>;
  ${Array.from({length: 16}, () => "ld.global.f32 %f1, [%rd1];").join("\n  ")}
  ${Array.from({length: 8},  () => "st.global.f32 [%rd2], %f1;").join("\n  ")}
  add.f32 %f2, %f1, %f1;
  add.f32 %f3, %f2, %f2;
  mul.f32 %f4, %f3, %f3;
  ret;
}
`;
    const ptx = featuresFromPtx(ptxMod, "_Z7smMemBK");
    const out = analyzeMemory(ptx);
    expect(out.class).toBe("memory_bound");
  });

  // ── Gap 4: Typed store bytes (STG.128 = 16 bytes, not 4) ───────────────────
  //
  // Before: all stores assumed 4 bytes → STG.128 gave 4× undercount of write BW.
  // After:  STG.128 → 16 bytes, STG.64 → 8 bytes, STG.32 → 4 bytes.

  it("stg128_stores_counted_as_16_bytes", () => {
    // 1 LDG.32 (4 B) + 1 STG.128 (16 B) → bytesMoved = 20 B.
    // Old: bytesMoved = 4 + 4 = 8 B (STG was assumed 4 bytes).
    let a = 0x3200;
    const lines = ["Function : _Z7stg128K", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);    a += 0x10;
    lines.push(`${sassHx(a)} STG.E.128 [R4], R6;`);   a += 0x10;
    const ptxMod =
      PTX_HEAD +
      `\n.visible .entry _Z7stg128K(.param .u64 p) { .reg .f32 %f<4>; ret; }\n`;
    const ptx  = featuresFromPtx(ptxMod, "_Z7stg128K");
    const sass = featuresFromSass(lines.join("\n"), "_Z7stg128K");
    // Direct SASS feature check
    expect(sass.stg_128).toBe(1);
    expect(sass.stg_32).toBe(0);
    const out = analyzeMemory(ptx, sass);
    // bytes_proxy must reflect 4 (LDG.32) + 16 (STG.128) = 20, NOT 4+4 = 8
    expect(out.bytes_proxy).toBe(20);
  });

  it("stg32_and_stg64_counted_correctly", () => {
    // Verify stg_64 and stg_32 classification in sass_features.
    let a = 0x3300;
    const lines = ["Function : _Z6stgWdK", ""];
    lines.push(`${sassHx(a)} STG.E.64 [R0], R2;`);  a += 0x10;  // 8 bytes
    lines.push(`${sassHx(a)} STG.E.32 [R4], R6;`);  a += 0x10;  // 4 bytes
    lines.push(`${sassHx(a)} STG.E.32 [R8], R10;`); a += 0x10;  // 4 bytes
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);               // 4 bytes
    const ptxMod =
      PTX_HEAD +
      `\n.visible .entry _Z6stgWdK(.param .u64 p) { .reg .f32 %f<4>; ret; }\n`;
    const ptx  = featuresFromPtx(ptxMod, "_Z6stgWdK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6stgWdK");
    expect(sass.stg_64).toBe(1);
    expect(sass.stg_32).toBe(2);
    const out = analyzeMemory(ptx, sass);
    // bytes: 4 (LDG.32) + 8 (STG.64) + 4×2 (STG.32×2) = 20
    expect(out.bytes_proxy).toBe(20);
  });

  // ── Gap 5: Streaming confidence bump ───────────────────────────────────────
  //
  // LD.CS (cache-streaming bypass) is an unambiguous memory-bound signal.
  // The confidence should be higher than the base SASS-available score.

  it("streaming_access_raises_confidence", () => {
    // LDG.E.CS loads → isStreaming = true, low intensity.
    let a = 0x3400;
    const lines = ["Function : _Z6csLoadK", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDG.E.CS.32 R${i}, [R${i + 8}];`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
    const ptxMod =
      PTX_HEAD +
      `\n.visible .entry _Z6csLoadK(.param .u64 p) { .reg .f32 %f<4>; ret; }\n`;
    const ptx  = featuresFromPtx(ptxMod, "_Z6csLoadK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6csLoadK");
    // Confirm streaming was detected
    expect(sass.cs_loads).toBeGreaterThan(0);
    const out = analyzeMemory(ptx, sass);
    expect(out.class).toBe("memory_bound");
    // Streaming bump: confidence > base SASS score of 0.85
    expect(out.confidence).toBeGreaterThan(0.85);
  });

  // ── Gap 6: Small-kernel confidence penalty ─────────────────────────────────
  //
  // A kernel with 2 global ops has too little data; confidence must be lower.

  it("small_kernel_has_lower_confidence_than_large", () => {
    // Large kernel: 20 loads + 10 stores = 30 global ops
    const buildPtx = (ops: number) =>
      PTX_HEAD +
      `\n.visible .entry _Zk(.param .u64 p) {\n  .reg .f32 %f<4>; .reg .u64 %rd<4>;\n  ${Array.from({length: ops}, () => "ld.global.f32 %f1, [%rd1];").join("\n  ")}\n  ret;\n}\n`;

    const smallPtx = featuresFromPtx(buildPtx(2), "_Zk");
    const largePtx = featuresFromPtx(buildPtx(20), "_Zk");

    const smallOut = analyzeMemory(smallPtx);
    const largeOut = analyzeMemory(largePtx);
    expect(smallOut.confidence).toBeLessThan(largeOut.confidence);
  });

  it("store_vectorization_score_reports_typed_store_width", () => {
    let a = 0x3500;
    const lines = ["Function : _Z6stVecK", ""];
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} STG.E.128 [R${i * 2}], R${i * 2 + 1};`);
      a += 0x10;
    }
    const ptxMod =
      PTX_HEAD +
      `\n.visible .entry _Z6stVecK(.param .u64 p) { .reg .f32 %f<4>; ret; }\n`;
    const ptx = featuresFromPtx(ptxMod, "_Z6stVecK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6stVecK");
    const out = analyzeMemory(ptx, sass);
    expect(out.store_vectorization_score).toBe(1);
  });

  it("load_store_balance_classifies_read_write_direction", () => {
    const readHeavyPtx =
      PTX_HEAD +
      `
.visible .entry _Z8readDomK(.param .u64 p) {
  .reg .f32 %f<4>; .reg .u64 %rd<4>;
  ld.global.f32 %f0, [%rd1];
  ld.global.f32 %f1, [%rd1];
  ld.global.f32 %f2, [%rd1];
  ld.global.f32 %f3, [%rd1];
  ld.global.f32 %f0, [%rd1];
  ld.global.f32 %f1, [%rd1];
  ld.global.f32 %f2, [%rd1];
  ld.global.f32 %f3, [%rd1];
  ld.global.f32 %f0, [%rd1];
  ld.global.f32 %f1, [%rd1];
  st.global.f32 [%rd2], %f0;
  ret;
}
`;
    const writeHeavyPtx =
      PTX_HEAD +
      `
.visible .entry _Z9writeDomK(.param .u64 p) {
  .reg .f32 %f<4>; .reg .u64 %rd<4>;
  ld.global.f32 %f0, [%rd1];
  st.global.f32 [%rd2], %f0;
  st.global.f32 [%rd2], %f0;
  st.global.f32 [%rd2], %f0;
  st.global.f32 [%rd2], %f0;
  st.global.f32 [%rd2], %f0;
  ret;
}
`;
    const readOut = analyzeMemory(featuresFromPtx(readHeavyPtx, "_Z8readDomK"));
    const writeOut = analyzeMemory(featuresFromPtx(writeHeavyPtx, "_Z9writeDomK"));
    expect(readOut.load_store_balance).toBe("read_dominated");
    expect(writeOut.load_store_balance).toBe("write_dominated");
  });

  // ── B4: compute_source provenance for zero-arithmetic SASS kernels ──────────
  //
  // When SASS is provided but contains no arithmetic instructions (e.g. a pure
  // memory-copy or control-flow stub), the old condition `sassFlopsProxy > 0`
  // evaluated to false and reported compute_source = "ptx", contradicting the
  // actual data source and producing misleading provenance in the UI.
  //
  // The fix: `sassFeatures !== undefined ? "sass" : "ptx"` always attributes to
  // the source that was actually present, regardless of count value.
  it("B4 – SASS with zero arithmetic ops still reports compute_source = 'sass'", () => {
    // Build a SASS-only kernel that moves data but does no arithmetic at all.
    let a = 0x8000;
    const lines: string[] = ["Function : _Z8memcpyKv", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDG.E.128 R${i * 4}, [R24];`);
      a += 0x10;
    }
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} STG.E.128 [R24], R${i * 4};`);
      a += 0x10;
    }
    // No FFMA / FADD / IMAD — arithmetic_ops = 0, sassFlopsProxy = 0.
    const sass = featuresFromSass(lines.join("\n"), "_Z8memcpyKv");
    const ptxEmpty = featuresFromPtx(
      PTX_HEAD + `\n.visible .entry _Z8memcpyKv(.param .u64 p)\n{\n  ret;\n}\n`,
      "_Z8memcpyKv"
    );

    const out = analyzeMemory(ptxEmpty, sass);
    // SASS was present → must say "sass", not "ptx"
    expect(out.compute_source).toBe("sass");
    // sass_flops_proxy should be 0 (no arithmetic), not null
    expect(out.sass_flops_proxy).toBe(0);
    // global memory source should also be "sass" (8 loads + 8 stores)
    expect(out.global_mem_source).toBe("sass");
    // Kernel should be memory-bound (loads + stores with zero compute)
    expect(out.class).toBe("memory_bound");
  });

  // ── B6: confidence override for globalMemOps = 0 must respect SASS presence ─
  //
  // Old code: `confidence = 0.30` unconditionally when globalMemOps === 0.
  // This discarded the +0.25 SASS bonus, so a SASS-verified pure-compute kernel
  // received the same low score as a PTX-only guess.
  //
  // New rule:
  //   - SASS present & zero global ops → SASS confirmed it → high confidence kept
  //   - SASS absent  & zero global ops → PTX zero is unreliable  → 0.30 override

  it("B6 – SASS-confirmed zero globalOps keeps high confidence (≥ 0.80)", () => {
    // Build a pure-compute SASS kernel: only FFMA, no LDG/STG at all.
    let a = 0xb000;
    const lines: string[] = ["Function : _Z6pureCoK", ""];
    for (let i = 0; i < 64; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const sass = featuresFromSass(lines.join("\n"), "_Z6pureCoK");
    const ptxEmpty = featuresFromPtx(
      PTX_HEAD + `\n.visible .entry _Z6pureCoK(.param .u64 p)\n{\n  ret;\n}\n`,
      "_Z6pureCoK"
    );

    const out = analyzeMemory(ptxEmpty, sass);
    expect(out.global_mem_ops).toBe(0);
    expect(out.class).toBe("compute_friendly");
    // SASS confirmed zero global ops → confidence must be high, not 0.30
    expect(out.confidence).toBeGreaterThanOrEqual(0.80);
  });

  it("B6 – PTX-only zero globalOps keeps low confidence (≤ 0.35)", () => {
    // PTX with no global loads or stores — PTX zero is unreliable.
    const ptxNoMem =
      PTX_HEAD +
      `
.visible .entry _Z8noMemPtxK(.param .u64 p)
{
  .reg .f32 %f<4>;
  fma.rn.f32 %f0, %f1, %f2, %f3;
  fma.rn.f32 %f0, %f1, %f2, %f3;
  ret;
}
`;
    const out = analyzeMemory(featuresFromPtx(ptxNoMem, "_Z8noMemPtxK"));
    expect(out.global_mem_ops).toBe(0);
    // PTX-only zero → hard to classify → confidence should stay ≤ 0.35
    expect(out.confidence).toBeLessThanOrEqual(0.35);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Gap 1 — FP64 FLOP-proxy weighting in memory_model
// (FP64 ops add an extra `fp64_arith_ops × 14` penalty so HPC kernels
//  aren't mis-classified as memory_bound)
// ─────────────────────────────────────────────────────────────────────────────

describe("memory_model — Gap 1: FP64 FLOP-proxy weighting", () => {
  it("fp64_heavy_kernel_lifts_flops_proxy_above_fp32_only", () => {
    // Build two kernels with the same memory footprint and same arithmetic_ops
    // count but different precisions: pure FP32 (FFMA) vs pure FP64 (DFMA).
    // The FP64 kernel should report a higher flops_proxy because of the +14
    // per-op penalty added in memory_model.
    const ptxStub =
      PTX_HEAD +
      `\n.visible .entry _Z6fp64K(.param .u64 p) { .reg .f32 %f<4>; ret; }\n`;
    const ptx = featuresFromPtx(ptxStub, "_Z6fp64K");

    let a32 = 0x3a00;
    const fp32Lines = ["Function : _Z6fp64K", ""];
    fp32Lines.push(`${sassHx(a32)} LDG.E.32 R0, [R2];`); a32 += 0x10;
    for (let i = 0; i < 8; i++) {
      fp32Lines.push(`${sassHx(a32)} FFMA.FTZ R0, R1, R2, R3;`);
      a32 += 0x10;
    }
    const sassFp32 = featuresFromSass(fp32Lines.join("\n"), "_Z6fp64K");

    let a64 = 0x3b00;
    const fp64Lines = ["Function : _Z6fp64K", ""];
    fp64Lines.push(`${sassHx(a64)} LDG.E.32 R0, [R2];`); a64 += 0x10;
    for (let i = 0; i < 8; i++) {
      fp64Lines.push(`${sassHx(a64)} DFMA R0, R2, R4, R6;`);
      a64 += 0x10;
    }
    const sassFp64 = featuresFromSass(fp64Lines.join("\n"), "_Z6fp64K");

    // arithmetic_ops should be identical (both 8) — FP64 ops also bump arithmetic_ops.
    expect(sassFp32.arithmetic_ops).toBe(sassFp64.arithmetic_ops);
    expect(sassFp64.fp64_arith_ops).toBe(8);
    expect(sassFp32.fp64_arith_ops).toBe(0);

    const fp32Out = analyzeMemory(ptx, sassFp32);
    const fp64Out = analyzeMemory(ptx, sassFp64);
    // The +14 penalty per FP64 op should produce 8 × 14 = 112 extra FLOPs.
    expect(fp64Out.flops_proxy).toBeGreaterThan(fp32Out.flops_proxy);
    expect(fp64Out.flops_proxy! - fp32Out.flops_proxy!).toBeCloseTo(8 * 14, 5);
  });

  it("fp64_kernel_arithmetic_intensity_higher_than_fp32_with_same_memory", () => {
    // Same memory footprint, FP64 should have higher arithmetic intensity.
    const ptxStub =
      PTX_HEAD +
      `\n.visible .entry _Z6fpInt(.param .u64 p) { .reg .f32 %f<4>; ret; }\n`;
    const ptx = featuresFromPtx(ptxStub, "_Z6fpInt");

    const buildSass = (op: string): string => {
      let a = 0x3c00;
      const lines = ["Function : _Z6fpInt", ""];
      // Equal memory traffic in both cases.
      for (let i = 0; i < 4; i++) {
        lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);
        a += 0x10;
      }
      for (let i = 0; i < 16; i++) {
        lines.push(`${sassHx(a)} ${op} R0, R1, R2, R3;`);
        a += 0x10;
      }
      return lines.join("\n");
    };

    const fp32 = analyzeMemory(ptx, featuresFromSass(buildSass("FFMA.FTZ"), "_Z6fpInt"));
    const fp64 = analyzeMemory(ptx, featuresFromSass(buildSass("DFMA"),    "_Z6fpInt"));
    expect(fp64.arithmetic_intensity_ops_per_byte!).toBeGreaterThan(
      fp32.arithmetic_intensity_ops_per_byte!
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Gap 6 — cache_policy precedence: dominant policy wins (cs vs cg)
// ─────────────────────────────────────────────────────────────────────────────

describe("memory_model — Gap 6: cache_policy dominance", () => {
  function buildSassCachePolicy(opts: {
    csCount: number;
    cgCount: number;
  }): string {
    let a = 0x4000;
    const lines = ["Function : _Z6cacheK", ""];
    for (let i = 0; i < opts.csCount; i++) {
      lines.push(`${sassHx(a)} LDG.E.CS.32 R${i % 8}, [R${(i % 4) + 8}];`);
      a += 0x10;
    }
    for (let i = 0; i < opts.cgCount; i++) {
      lines.push(`${sassHx(a)} LDG.E.CG.32 R${i % 8}, [R${(i % 4) + 8}];`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
    return lines.join("\n");
  }

  function ptxStub(): import("../src/analyzer/ptx_features").PtxInstructionFeatures {
    return featuresFromPtx(
      PTX_HEAD +
        `\n.visible .entry _Z6cacheK(.param .u64 p) { .reg .f32 %f<4>; ret; }\n`,
      "_Z6cacheK"
    );
  }

  it("cs_dominates_cg_yields_streaming_even_with_one_cg_load", () => {
    // 1 CG + 16 CS — old code returned "L2" (any CG suppressed streaming).
    // After Gap 6 the dominant CS policy should win, yielding "streaming".
    const sass = featuresFromSass(
      buildSassCachePolicy({ csCount: 16, cgCount: 1 }),
      "_Z6cacheK"
    );
    expect(sass.cs_loads).toBe(16);
    expect(sass.cg_loads).toBe(1);
    const out = analyzeMemory(ptxStub(), sass);
    expect(out.cache_policy).toBe("streaming");
  });

  it("cg_dominates_cs_yields_L2", () => {
    // 12 CG + 1 CS — CG dominates; result must be "L2".
    const sass = featuresFromSass(
      buildSassCachePolicy({ csCount: 1, cgCount: 12 }),
      "_Z6cacheK"
    );
    const out = analyzeMemory(ptxStub(), sass);
    expect(out.cache_policy).toBe("L2");
  });

  it("equal_cs_and_cg_counts_yield_mixed", () => {
    // Tied counts → can't cleanly attribute → "mixed".
    const sass = featuresFromSass(
      buildSassCachePolicy({ csCount: 4, cgCount: 4 }),
      "_Z6cacheK"
    );
    expect(sass.cs_loads).toBe(4);
    expect(sass.cg_loads).toBe(4);
    const out = analyzeMemory(ptxStub(), sass);
    expect(out.cache_policy).toBe("mixed");
  });

  it("zero_typed_loads_yields_null_cache_policy", () => {
    // Plain LDG with no .CS / .CG modifier → no cache hint.
    let a = 0x4500;
    const lines = ["Function : _Z6cacheK", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R${i}, [R${i + 8}];`);
      a += 0x10;
    }
    const sass = featuresFromSass(lines.join("\n"), "_Z6cacheK");
    expect(sass.cs_loads).toBe(0);
    expect(sass.cg_loads).toBe(0);
    const out = analyzeMemory(ptxStub(), sass);
    expect(out.cache_policy).toBeNull();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Gap 2 — Sub-32-bit loads/stores feed estimateSassBytes() at correct widths.
//
// Before Gap 2, half/bf16 loads (LDG.E.U16, LDG.E.F16) and INT8/FP8 loads
// (LDG.E.U8) fell into the unknown-width fallback bucket and were costed at
// 4 bytes each.  This inflated bandwidth on FP16/INT8 kernels by 2–4× and
// pushed arithmetic intensity below thresholds → false memory_bound calls.
// ─────────────────────────────────────────────────────────────────────────────

describe("memory_model — Gap 2: sub-32-bit byte accounting", () => {
  function ptxStubFor(kernel: string): import("../src/analyzer/ptx_features").PtxInstructionFeatures {
    return featuresFromPtx(
      PTX_HEAD +
        `\n.visible .entry ${kernel}(.param .u64 p) { .reg .f32 %f<4>; ret; }\n`,
      kernel
    );
  }

  it("ldg_16_loads_cost_2_bytes_each_not_4", () => {
    // 8 × LDG.E.U16 → bytes_proxy must be 8 × 2 = 16 (not 8 × 4 = 32).
    let a = 0x5000;
    const lines = ["Function : _Z6sub16K", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDG.E.U16 R${i}, [R${i + 8}];`);
      a += 0x10;
    }
    const sass = featuresFromSass(lines.join("\n"), "_Z6sub16K");
    expect(sass.ldg_16).toBe(8);
    expect(sass.ldg_32).toBe(0);
    const out = analyzeMemory(ptxStubFor("_Z6sub16K"), sass);
    expect(out.bytes_proxy).toBe(16);
  });

  it("ldg_8_loads_cost_1_byte_each_not_4", () => {
    // 16 × LDG.E.U8 → bytes_proxy must be 16 × 1 = 16 (not 16 × 4 = 64).
    let a = 0x5100;
    const lines = ["Function : _Z5int8K", ""];
    for (let i = 0; i < 16; i++) {
      lines.push(`${sassHx(a)} LDG.E.U8 R${i}, [R${i + 16}];`);
      a += 0x10;
    }
    const sass = featuresFromSass(lines.join("\n"), "_Z5int8K");
    expect(sass.ldg_8).toBe(16);
    expect(sass.ldg_32).toBe(0);
    const out = analyzeMemory(ptxStubFor("_Z5int8K"), sass);
    expect(out.bytes_proxy).toBe(16);
  });

  it("stg_16_stores_cost_2_bytes_each_not_4", () => {
    // 4 × STG.E.U16 → bytes_proxy must be 4 × 2 = 8.
    let a = 0x5200;
    const lines = ["Function : _Z7stg16wK", ""];
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} STG.E.U16 [R${i * 2}], R${i * 2 + 1};`);
      a += 0x10;
    }
    const sass = featuresFromSass(lines.join("\n"), "_Z7stg16wK");
    expect(sass.stg_16).toBe(4);
    expect(sass.stg_32).toBe(0);
    const out = analyzeMemory(ptxStubFor("_Z7stg16wK"), sass);
    expect(out.bytes_proxy).toBe(8);
  });

  it("stg_8_stores_cost_1_byte_each_not_4", () => {
    let a = 0x5300;
    const lines = ["Function : _Z6stg8wK", ""];
    for (let i = 0; i < 6; i++) {
      lines.push(`${sassHx(a)} STG.E.U8 [R${i * 2}], R${i * 2 + 1};`);
      a += 0x10;
    }
    const sass = featuresFromSass(lines.join("\n"), "_Z6stg8wK");
    expect(sass.stg_8).toBe(6);
    expect(sass.stg_32).toBe(0);
    const out = analyzeMemory(ptxStubFor("_Z6stg8wK"), sass);
    expect(out.bytes_proxy).toBe(6);
  });

  it("fp16_kernel_arithmetic_intensity_higher_than_pre_gap2_estimate", () => {
    // Mixed: 16 × LDG.E.U16 (32 bytes after fix; 64 bytes before) + 16 × HFMA.
    // After Gap 2 the byte count is halved, so arithmetic intensity should be
    // measurably higher than the same workload reinterpreted as 32-bit loads.
    const ptx = ptxStubFor("_Z5fp16K");

    const buildSass = (loadOp: string): string => {
      let a = 0x5400;
      const lines = ["Function : _Z5fp16K", ""];
      for (let i = 0; i < 16; i++) {
        lines.push(`${sassHx(a)} ${loadOp} R${i}, [R${i + 16}];`);
        a += 0x10;
      }
      for (let i = 0; i < 16; i++) {
        lines.push(`${sassHx(a)} HFMA R${i % 8}, R0, R1, R2;`);
        a += 0x10;
      }
      return lines.join("\n");
    };

    const fp16 = analyzeMemory(ptx, featuresFromSass(buildSass("LDG.E.U16"), "_Z5fp16K"));
    const fp32 = analyzeMemory(ptx, featuresFromSass(buildSass("LDG.E.32"),  "_Z5fp16K"));

    // Same arithmetic side, half the bytes — fp16 must have higher intensity.
    expect(fp16.arithmetic_intensity_ops_per_byte!).toBeGreaterThan(
      fp32.arithmetic_intensity_ops_per_byte!
    );
    // Concretely: fp16 bytes = 32, fp32 bytes = 64 → ratio ≈ 2×.
    expect(fp16.bytes_proxy).toBe(32);
    expect(fp32.bytes_proxy).toBe(64);
  });

  it("constant_loads_do_NOT_inflate_bytes_proxy_for_global_traffic", () => {
    // LDC pulls from the constant bank — it must not be added to global byte
    // accounting.  bytes_proxy should still be 0 for a kernel that only does
    // LDC and arithmetic.
    let a = 0x5500;
    const lines = ["Function : _Z5constK", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDC R${i}, c[0x0][0x${(i * 4).toString(16)}];`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
    const sass = featuresFromSass(lines.join("\n"), "_Z5constK");
    expect(sass.const_loads).toBe(8);
    expect(sass.global_loads).toBe(0); // crucially: constant ≠ global
    const out = analyzeMemory(ptxStubFor("_Z5constK"), sass);
    // Pure compute (1 FFMA) with no global traffic → bytes_proxy is 0.
    expect(out.bytes_proxy).toBe(0);
  });
});
