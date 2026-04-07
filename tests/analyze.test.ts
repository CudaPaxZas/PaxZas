/**
 * Integration smoke: full async pipeline (mirrors tests/test_pipeline.py intent).
 */
import { describe, expect, it } from "vitest";
import { analyze } from "../src/analyzer/analyze";

const SAMPLE_PTX = `
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
ret;
}
`;

describe("analyze()", () => {
  it("runs PTX pipeline without error", async () => {
    const r = await analyze(SAMPLE_PTX, {}, undefined, undefined, "ampere-like-default");
    expect(r.error).toBeUndefined();
    expect(r.kind).toBe("ptx");
    expect(r.pattern).toBeDefined();
    expect(r.memory).toBeDefined();
    expect(r.occupancy_model).toBeDefined();
    expect(r.ptx_kernel).toBe("_Z3foov");
  });
});
