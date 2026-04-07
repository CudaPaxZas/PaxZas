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
