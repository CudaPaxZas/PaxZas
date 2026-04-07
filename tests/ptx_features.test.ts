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
});
