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
});
