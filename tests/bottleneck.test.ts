import { describe, expect, it } from "vitest";
import { heuristicBottleneck } from "../src/analyzer/bottleneck";
import { emptyInstructionFeatures } from "../src/analyzer/ptx_features";
import { featuresFromSass, sassHx, PTX_HEAD } from "./helpers/ir_fixtures";
import { featuresFromPtx } from "./helpers/ir_fixtures";

describe("heuristicBottleneck — I11 SASS parity", () => {
  it("uses_sass_tensor_flops_when_ptx_is_empty", () => {
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
    const sass = featuresFromSass(lines.join("\n"), "_Z6tensorK");
    const instr = emptyInstructionFeatures();
    const r = heuristicBottleneck(instr, 32, 8, 64, sass);
    expect(r.flops_proxy).toBeGreaterThan(1000);
    expect(r.notes).toContain("sass_flops_proxy");
    expect(r.notes).toContain("sass_global_bytes_proxy");
    expect(r.bottleneck).toBe("compute-bound");
  });

  it("ptx_only_still_works_without_sass", () => {
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z3add(.param .u64 p)
{
  .reg .f32 %f<4>;
  .reg .u64 %rd<4>;
  ld.global.f32 %f1, [%rd1];
  ld.global.f32 %f2, [%rd2];
  add.f32 %f3, %f1, %f2;
  st.global.f32 [%rd2], %f3;
  ret;
}
`;
    const ptx = featuresFromPtx(ptxMod, "_Z3add");
    const r = heuristicBottleneck(ptx, 32);
    expect(r.global_mem_ops).toBeGreaterThan(0);
    expect(r.notes).not.toContain("sass_flops_proxy");
  });
});
