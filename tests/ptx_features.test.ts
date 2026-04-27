/**
 * Mirrors tests/test_ptx_features.py
 */
import { describe, expect, it } from "vitest";
import {
  extractInstructionFeatures,
  findKernelBody,
  flopsHeuristic,
} from "../src/analyzer/ptx_features";

const SAMPLE = `
.version 8.0
.target sm_80
.visible .entry _Z3foov(
.param .u64 _Z3foov_param_0
)
.maxntid 256, 1, 1
.maxnreg 48
{
.reg .pred %p<2>;
.reg .f32 %f<4>;
ld.global.f32 %f1, [%rd1];
st.global.f32 [%rd2], %f2;
fma.rn.f32 %f3, %f1, %f2, %f1;
add.f32 %f4, %f3, %f1;
mul.f32 %f5, %f4, %f2;
bar.sync 0;
ret;
}
`;

describe("ptx_features (Python parity)", () => {
  it("find_kernel_body_and_features", () => {
    const [name, body] = findKernelBody(SAMPLE, undefined);
    expect(name).toBeDefined();
    expect(body).toBeDefined();
    const f = extractInstructionFeatures(body!);
    expect(f.global_loads).toBeGreaterThanOrEqual(1);
    expect(f.global_stores).toBeGreaterThanOrEqual(1);
    expect(f.fma).toBeGreaterThanOrEqual(1);
    expect(f.add).toBeGreaterThanOrEqual(1);
    expect(f.mul).toBeGreaterThanOrEqual(1);
    expect(f.barrier).toBeGreaterThanOrEqual(1);
    expect(f.reg_decl_lines).toBeGreaterThanOrEqual(2);
    expect(flopsHeuristic(f)).toBeGreaterThanOrEqual(4.0);
  });

  it("B1 – compound opcodes (red.add / atom.add / shfl.sync) must not inflate add/mul", () => {
    // A PTX body that contains compound opcodes which embed "add." or "mul."
    // as a sub-component.  Only the true scalar instructions should be counted.
    const body = `
      .reg .u32 %r<4>;
      .reg .f32 %f<4>;
      .reg .u64 %rd<4>;

      // These must NOT be counted as add/mul:
      red.add.s32   [%rd0], 1;
      red.add.f32   [%rd0], %f1;
      atom.add.s64  %r0, [%rd1], 1;
      atom.add.f32  %f2, [%rd1], %f1;
      shfl.sync.idx.b32 %r1, %r2, 0, 0x1f, %p0;

      // These MUST be counted:
      add.f32  %f3, %f1, %f2;
      add.s32  %r3, %r1, %r2;
      mul.f32  %f3, %f1, %f2;
      mul.wide.u32 %rd1, %r1, %r2;
    `;
    const f = extractInstructionFeatures(body);
    // Exactly 2 stand-alone add instructions
    expect(f.add).toBe(2);
    // Exactly 2 stand-alone mul instructions
    expect(f.mul).toBe(2);
  });

  it("I10 – counts backward loops for BB-style and dotted labels", () => {
    const body = `
BB0_1:
  add.s32 %r1, %r1, 1;
  bra BB0_1;
.Ltmp0:
  add.s32 %r2, %r2, 1;
  bra .Ltmp0;
`;
    const f = extractInstructionFeatures(body);
    expect(f.branches).toBe(2);
    expect(f.loops).toBe(2);
  });
});
