/**
 * Mirrors tests/test_pattern_model.py (same IR fixtures / assertions).
 */
import { describe, expect, it } from "vitest";
import { analyzePattern } from "../src/analyzer/pattern_model";
import {
  PTX_HEAD,
  featuresFromPtx,
  featuresFromSass,
  sassHx,
} from "./helpers/ir_fixtures";

function sassTiledLike(): string {
  const lines: string[] = ["Function : _Z5k_tiled", ""];
  let a = 0x0100;
  lines.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
  a += 0x10;
  for (let i = 0; i < 120; i++) {
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
    a += 0x10;
  }
  for (let i = 0; i < 20; i++) {
    lines.push(`${sassHx(a)} LDS.128 R8, [R4];`);
    a += 0x10;
  }
  for (let i = 0; i < 10; i++) {
    lines.push(`${sassHx(a)} STS.128 [R4], R8;`);
    a += 0x10;
  }
  lines.push(`${sassHx(a)} BAR.SYNC 0;`);
  a += 0x10;
  lines.push(`${sassHx(a)} BAR.SYNC 0;`);
  return lines.join("\n");
}

function sassReductionLike(): string {
  const lines: string[] = ["Function : _Z5k_red", ""];
  let a = 0x0200;
  for (let i = 0; i < 14; i++) {
    lines.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
    a += 0x10;
  }
  for (let i = 0; i < 4; i++) {
    lines.push(`${sassHx(a)} STG.E.32 [R2], R4;`);
    a += 0x10;
  }
  for (let i = 0; i < 10; i++) {
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
    a += 0x10;
  }
  for (let i = 0; i < 10; i++) {
    lines.push(`${sassHx(a)} LDS.128 R8, [R4];`);
    a += 0x10;
  }
  for (let i = 0; i < 8; i++) {
    lines.push(`${sassHx(a)} STS.128 [R4], R8;`);
    a += 0x10;
  }
  lines.push(`${sassHx(a)} BAR.SYNC 0;`);
  a += 0x10;
  lines.push(`${sassHx(a)} BAR.SYNC 0;`);
  return lines.join("\n");
}

function ptxOneLoop(): string {
  return (
    PTX_HEAD +
    `
.visible .entry _Z5k_loop(
  .param .u64 p
)
{
  .reg .pred %p<2>;
  .reg .u32 %r<4>;
  .reg .f32 %f<4>;
  .reg .u64 %rd<4>;

  mov.u32 %r0, 0;
  mov.u32 %r1, 4;
  mov.u64 %rd1, 0;

$L_loop:
  ld.global.f32 %f0, [%rd1];
  add.f32 %f1, %f0, %f0;
  add.u32 %r0, %r0, 1;
  setp.lt.u32 %p0, %r0, %r1;
  @%p0 bra $L_loop;

  ret;
}
`
  );
}

function ptxTwoLoops(): string {
  return (
    PTX_HEAD +
    `
.visible .entry _Z8k_2loops(
  .param .u64 p
)
{
  .reg .pred %p<2>;
  .reg .u32 %r<4>;
  .reg .f32 %f<4>;
  .reg .u64 %rd<4>;

  mov.u32 %r0, 0;
  mov.u32 %r1, 2;
  mov.u64 %rd1, 0;

$L_a:
  ld.global.f32 %f0, [%rd1];
  add.u32 %r0, %r0, 1;
  setp.lt.u32 %p0, %r0, %r1;
  @%p0 bra $L_a;

  mov.u32 %r0, 0;

$L_b:
  ld.global.f32 %f1, [%rd1];
  add.u32 %r0, %r0, 1;
  setp.lt.u32 %p1, %r0, %r1;
  @%p1 bra $L_b;

  ret;
}
`
  );
}

describe("pattern_model (Python parity)", () => {
  it("elementwise_from_low_control_signals", () => {
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z4elem(
  .param .u64 p
)
{
  .reg .f32 %f<4>;
  .reg .u64 %rd<4>;
  ld.global.f32 %f1, [%rd1];
  st.global.f32 [%rd2], %f1;
  add.f32 %f2, %f1, %f1;
  mul.f32 %f3, %f2, %f2;
  ret;
}
`;
    const ptx = featuresFromPtx(ptxMod, "_Z4elem");
    const out = analyzePattern(ptx);
    expect(["elementwise", "irregular"]).toContain(out.class);
    expect(out.source).toBe("ptx");
  });

  it("tiled_with_shared_and_barrier", () => {
    const ptx = featuresFromPtx(ptxOneLoop(), "_Z5k_loop");
    const sass = featuresFromSass(sassTiledLike(), "_Z5k_tiled");
    const out = analyzePattern(ptx, sass);
    expect(out.class).toBe("tiled");
    expect(out.source).toBe("sass");
    expect(out.shared_to_global_ratio).toBeGreaterThan(1.5);
    expect(out.compute_to_memory_ratio).toBeGreaterThan(8.0);
    expect(ptx.loops).toBeGreaterThanOrEqual(1);
  });

  it("reduction_shared_barrier_loops_low_compute_intensity", () => {
    const ptx = featuresFromPtx(ptxOneLoop(), "_Z5k_loop");
    const sass = featuresFromSass(sassReductionLike(), "_Z5k_red");
    const out = analyzePattern(ptx, sass);
    expect(out.class).toBe("reduction");
    expect(out.compute_to_memory_ratio).toBeLessThanOrEqual(8.0);
  });

  it("compute_heavy_no_shared_no_barrier", () => {
    const fmaBlock = Array.from(
      { length: 40 },
      () => "fma.rn.f32 %f1, %f2, %f3, %f4;"
    ).join("\n  ");
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z3heavy(
  .param .u64 p
)
{
  .reg .f32 %f<8>;
  .reg .u64 %rd<4>;
  mov.f32 %f2, 0f3F800000;
  mov.f32 %f3, 0f3F800000;
  mov.f32 %f4, 0f00000000;
  ${fmaBlock}
  ret;
}
`;
    const ptx = featuresFromPtx(ptxMod, "_Z3heavy");
    let a = 0x0400;
    const heavy: string[] = ["Function : _Z3heavy", ""];
    heavy.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
    a += 0x10;
    heavy.push(`${sassHx(a)} STG.E.32 [R2], R4;`);
    a += 0x10;
    for (let i = 0; i < 50; i++) {
      heavy.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const sassMod = heavy.join("\n");
    const sass = featuresFromSass(sassMod, "_Z3heavy");
    const out = analyzePattern(ptx, sass);
    expect(out.class).toBe("compute_heavy");
    expect(out.compute_to_memory_ratio).toBeGreaterThanOrEqual(2.0);
  });

  it("control_heavy_with_high_branch_density", () => {
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z3ctl(
  .param .u64 p
)
{
  .reg .u64 %rd<4>;
  ret;
}
`;
    const ptx = featuresFromPtx(ptxMod, "_Z3ctl");
    let ax = 0x0500;
    const ctl: string[] = ["Function : _Z3ctl", ""];
    ctl.push(`${sassHx(ax)} LDG.E.32 R4, [R2];`);
    ax += 0x10;
    ctl.push(`${sassHx(ax)} STG.E.32 [R2], R4;`);
    ax += 0x10;
    ctl.push(`${sassHx(ax)} FFMA.FTZ R0, R1, R2, R3;`);
    ax += 0x10;
    ctl.push(`${sassHx(ax)} FFMA.FTZ R0, R1, R2, R3;`);
    ax += 0x10;
    for (let i = 0; i < 20; i++) {
      ctl.push(`${sassHx(ax)} BRA 0x${i.toString(16)};`);
      ax += 0x10;
    }
    const sass = featuresFromSass(ctl.join("\n"), "_Z3ctl");
    const out = analyzePattern(ptx, sass);
    expect(out.class).toBe("control_heavy");
    expect(out.branch_density).toBeGreaterThan(0.1);
  });

  it("streaming_detection_and_top_level_fields", () => {
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z4str(
  .param .u64 p
)
{
  .reg .f32 %f<4>;
  .reg .u64 %rd<4>;
  ld.global.f32 %f0, [%rd1];
  ld.global.f32 %f1, [%rd1];
  ld.global.f32 %f2, [%rd1];
  ld.global.f32 %f3, [%rd1];
  ld.global.f32 %f0, [%rd1];
  ld.global.f32 %f1, [%rd1];
  ld.global.f32 %f2, [%rd1];
  ld.global.f32 %f3, [%rd1];
  st.global.f32 [%rd2], %f0;
  st.global.f32 [%rd2], %f1;
  st.global.f32 [%rd2], %f2;
  st.global.f32 [%rd2], %f3;
  add.f32 %f0, %f1, %f2;
  mul.f32 %f1, %f0, %f0;
  ret;
}
`;
    const ptx = featuresFromPtx(ptxMod, "_Z4str");
    const out = analyzePattern(ptx);
    expect(out.streaming).toBe(true);
    expect(out.streaming).toBe(true);
    expect(out.sync_efficiency).toBe("inefficient");
    expect(out.interleaving).toBe("mixed");
  });

  it("sync_efficiency_buckets", () => {
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z4syn(
  .param .u64 p
)
{ ret; }
`;
    const ptx = featuresFromPtx(ptxMod, "_Z4syn");

    let a = 0x0600;
    const ineffLines: string[] = ["Function : _Z4syn", ""];
    ineffLines.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
    a += 0x10;
    ineffLines.push(`${sassHx(a)} STG.E.32 [R2], R4;`);
    a += 0x10;
    for (let i = 0; i < 3; i++) {
      ineffLines.push(`${sassHx(a)} BAR.SYNC 0;`);
      a += 0x10;
    }
    for (let i = 0; i < 10; i++) {
      ineffLines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const outInefficient = analyzePattern(
      ptx,
      featuresFromSass(ineffLines.join("\n"), "_Z4syn")
    );
    expect(outInefficient.sync_efficiency).toBe("inefficient");

    a = 0x0700;
    const modLines: string[] = ["Function : _Z4syn", ""];
    modLines.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
    a += 0x10;
    modLines.push(`${sassHx(a)} STG.E.32 [R2], R4;`);
    a += 0x10;
    for (let i = 0; i < 2; i++) {
      modLines.push(`${sassHx(a)} BAR.SYNC 0;`);
      a += 0x10;
    }
    for (let i = 0; i < 100; i++) {
      modLines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const outModerate = analyzePattern(
      ptx,
      featuresFromSass(modLines.join("\n"), "_Z4syn")
    );
    expect(outModerate.sync_efficiency).toBe("moderate");

    a = 0x0800;
    const effLines: string[] = ["Function : _Z4syn", ""];
    effLines.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
    a += 0x10;
    effLines.push(`${sassHx(a)} STG.E.32 [R2], R4;`);
    a += 0x10;
    effLines.push(`${sassHx(a)} BAR.SYNC 0;`);
    a += 0x10;
    for (let i = 0; i < 220; i++) {
      effLines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const outEfficient = analyzePattern(
      ptx,
      featuresFromSass(effLines.join("\n"), "_Z4syn")
    );
    expect(outEfficient.sync_efficiency).toBe("efficient");
  });

  it("interleaving_detects_interleaved_and_stacked", () => {
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z3ilv(
  .param .u64 p
)
{ ret; }
`;
    const ptx = featuresFromPtx(ptxMod, "_Z3ilv");
    const interleaved: string[] = ["Function : _Z3ilv", ""];
    let a = 0x0900;
    for (let i = 0; i < 6; i++) {
      interleaved.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
      a += 0x10;
      interleaved.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const sassI = interleaved.join("\n");
    const outInterleaved = analyzePattern(ptx, featuresFromSass(sassI, "_Z3ilv"));
    expect(outInterleaved.interleaving).toBe("interleaved");
    expect(outInterleaved.interleaving).toBe("interleaved");
    expect(featuresFromSass(sassI, "_Z3ilv").instruction_sequence.length).toBeGreaterThanOrEqual(
      4
    );

    const stacked: string[] = ["Function : _Z3stk", ""];
    a = 0x0a00;
    for (let i = 0; i < 5; i++) {
      stacked.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
      a += 0x10;
    }
    stacked.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
    const sassS = stacked.join("\n");
    const ptx2 = featuresFromPtx(
      PTX_HEAD +
        `
.visible .entry _Z3stk(
  .param .u64 p
)
{ ret; }
`,
      "_Z3stk"
    );
    const outStacked = analyzePattern(ptx2, featuresFromSass(sassS, "_Z3stk"));
    expect(outStacked.interleaving).toBe("stacked");
    expect(outStacked.interleaving).toBe("stacked");
  });

  it("micro_high_looping_and_sync_heavy", () => {
    const ptx = featuresFromPtx(ptxTwoLoops(), "_Z8k_2loops");
    let a = 0x0b00;
    const mlns: string[] = ["Function : _Z8k_2loops", ""];
    mlns.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
    a += 0x10;
    mlns.push(`${sassHx(a)} STG.E.32 [R2], R4;`);
    a += 0x10;
    for (let i = 0; i < 2; i++) {
      mlns.push(`${sassHx(a)} LDS.32 R8, [R4];`);
      a += 0x10;
    }
    mlns.push(`${sassHx(a)} STS.32 [R4], R8;`);
    a += 0x10;
    for (let i = 0; i < 2; i++) {
      mlns.push(`${sassHx(a)} BAR.SYNC 0;`);
      a += 0x10;
    }
    for (let i = 0; i < 8; i++) {
      mlns.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const sass = featuresFromSass(mlns.join("\n"), "_Z8k_2loops");
    const out = analyzePattern(ptx, sass);
    expect(out.high_looping).toBe(true);
    expect(out.sync_heavy).toBe(true);
  });

  it("micro_control_irregular_and_control_dominated", () => {
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z3mic(
  .param .u64 p
)
{ ret; }
`;
    const ptx = featuresFromPtx(ptxMod, "_Z3mic");
    let a = 0x0c00;
    const mic: string[] = ["Function : _Z3mic", ""];
    mic.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
    a += 0x10;
    mic.push(`${sassHx(a)} STG.E.32 [R2], R4;`);
    a += 0x10;
    for (let i = 0; i < 8; i++) {
      mic.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    for (let i = 0; i < 6; i++) {
      mic.push(`${sassHx(a)} BRA 0x${i.toString(16)};`);
      a += 0x10;
    }
    const sass = featuresFromSass(mic.join("\n"), "_Z3mic");
    const out = analyzePattern(ptx, sass);
    expect(out.control_irregular).toBe(true);
    expect(out.control_dominated).toBe(true);
  });

  it("micro_complex_kernel_and_tensor_dominated", () => {
    const ptx = featuresFromPtx(ptxOneLoop(), "_Z5k_loop");
    let a = 0x0d00;
    const cpx: string[] = ["Function : _Z5k_loop", ""];
    cpx.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
    a += 0x10;
    for (let i = 0; i < 4; i++) {
      cpx.push(`${sassHx(a)} LDS.128 R8, [R4];`);
      a += 0x10;
    }
    for (let i = 0; i < 4; i++) {
      cpx.push(`${sassHx(a)} STS.128 [R4], R8;`);
      a += 0x10;
    }
    cpx.push(`${sassHx(a)} BAR.SYNC 0;`);
    a += 0x10;
    for (let i = 0; i < 20; i++) {
      cpx.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    for (let i = 0; i < 4; i++) {
      cpx.push(`${sassHx(a)} HMMA.16816.F16 R4, R2, R8, R4;`);
      a += 0x10;
    }
    const sass = featuresFromSass(cpx.join("\n"), "_Z5k_loop");
    const out = analyzePattern(ptx, sass);
    expect(out.complex_kernel).toBe(true);
    expect(out.tensor_dominated).toBe(true);
  });
});

describe("pattern_model — inefficiency signals", () => {
  // ── spill_risk ──────────────────────────────────────────────────────────────

  it("spill_risk_true_when_ldl_stl_present", () => {
    // LDL/STL in SASS proves register-file spilling; spill_risk must be true.
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z6spillK(.param .u64 p) {
  .reg .f32 %f<4>;
  .reg .u64 %rd<4>;
  ld.global.f32 %f1, [%rd1];
  ret;
}
`;
    let a = 0x0000;
    const lines = ["Function : _Z6spillK", ""];
    lines.push(`${sassHx(a)} LDL R0, [R2+0x10];`);   a += 0x10; // spill restore
    lines.push(`${sassHx(a)} STL [R4+0x20], R6;`);   a += 0x10; // spill save
    lines.push(`${sassHx(a)} LDG.E.32 R8, [R10];`);  a += 0x10;
    lines.push(`${sassHx(a)} FFMA.RN R0, R1, R2, R0;`);
    const ptx  = featuresFromPtx(ptxMod, "_Z6spillK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6spillK");
    const out  = analyzePattern(ptx, sass);
    expect(out.spill_risk).toBe(true);
    expect(out.spill_risk).toBe(true);
  });

  it("spill_risk_false_when_no_ldl_stl", () => {
    // Without LDL/STL, spill_risk must be false even when register count is high.
    let a = 0x0000;
    const lines = ["Function : _Z6cleanK", ""];
    for (let i = 0; i < 5; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R${i}, [R${i + 10}];`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} FFMA.RN R0, R1, R2, R0;`);
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z6cleanK(.param .u64 p) {
  .reg .f32 %f<4>; .reg .u64 %rd<4>;
  ld.global.f32 %f1, [%rd1]; ret;
}
`;
    const ptx  = featuresFromPtx(ptxMod, "_Z6cleanK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6cleanK");
    const out  = analyzePattern(ptx, sass);
    expect(out.spill_risk).toBe(false);
  });

  // ── uncoalesced_risk ────────────────────────────────────────────────────────

  it("uncoalesced_risk_true_when_narrow_loads_dominate", () => {
    // 8 narrow 32-bit loads and only 1 wide 128-bit load → 88% narrow ratio
    // (> 75% threshold), so uncoalesced_risk should be true.
    let a = 0x0100;
    const lines = ["Function : _Z7uncoalK", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R${i * 2}, [R${i * 2 + 1}];`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} LDG.E.128 R0, [R1];`);  a += 0x10; // only wide load
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} FFMA.RN R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z7uncoalK(.param .u64 p) {
  .reg .f32 %f<4>; .reg .u64 %rd<4>;
  ld.global.f32 %f1, [%rd1]; ret;
}
`;
    const ptx  = featuresFromPtx(ptxMod, "_Z7uncoalK");
    const sass = featuresFromSass(lines.join("\n"), "_Z7uncoalK");
    const out  = analyzePattern(ptx, sass);
    expect(out.uncoalesced_risk).toBe(true);
    expect(out.uncoalesced_risk).toBe(true);
  });

  it("uncoalesced_risk_false_when_wide_loads_dominate", () => {
    // Mostly 128-bit loads → well-coalesced, uncoalesced_risk must be false.
    let a = 0x0200;
    const lines = ["Function : _Z6coalK", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDG.E.128 R${i * 4}, [R${i * 4 + 16}];`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R1];`);  // only one narrow load
    a += 0x10;
    lines.push(`${sassHx(a)} FFMA.RN R0, R1, R2, R3;`);
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z6coalK(.param .u64 p) {
  .reg .f32 %f<4>; .reg .u64 %rd<4>;
  ld.global.f32 %f1, [%rd1]; ret;
}
`;
    const ptx  = featuresFromPtx(ptxMod, "_Z6coalK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6coalK");
    const out  = analyzePattern(ptx, sass);
    expect(out.uncoalesced_risk).toBe(false);
  });

  // ── missing_tensor_cores ────────────────────────────────────────────────────

  it("missing_tensor_cores_true_for_ffma_heavy_no_mma", () => {
    // High compute/memory ratio with many FFMA but zero MMA/HMMA should flag
    // the missing tensor-core opportunity.
    let a = 0x0300;
    const lines = ["Function : _Z8gemmFFMAK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);  a += 0x10;
    for (let i = 0; i < 40; i++) {
      lines.push(`${sassHx(a)} FFMA.RN R${i % 8}, R${(i+1)%8}, R${(i+2)%8}, R${(i+3)%8};`);
      a += 0x10;
    }
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z8gemmFFMAK(.param .u64 p) {
  .reg .f32 %f<8>; .reg .u64 %rd<4>;
  ${Array.from({length: 40}, () => "fma.rn.f32 %f1, %f2, %f3, %f4;").join("\n  ")}
  ret;
}
`;
    const ptx  = featuresFromPtx(ptxMod, "_Z8gemmFFMAK");
    const sass = featuresFromSass(lines.join("\n"), "_Z8gemmFFMAK");
    const out  = analyzePattern(ptx, sass);
    expect(out.uses_tensor_cores).toBe(false);
    expect(out.missing_tensor_cores).toBe(true);
    expect(out.missing_tensor_cores).toBe(true);
  });

  it("missing_tensor_cores_false_when_hmma_present", () => {
    // Once even one HMMA is present, missing_tensor_cores must be false.
    let a = 0x0400;
    const lines = ["Function : _Z8gemmHMMAK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);  a += 0x10;
    for (let i = 0; i < 20; i++) {
      lines.push(`${sassHx(a)} HMMA.16816.F32 {R0,R1,R2,R3},{R4,R5},{R6,R7},{R0,R1,R2,R3};`);
      a += 0x10;
    }
    const ptxMod =
      PTX_HEAD + `\n.visible .entry _Z8gemmHMMAK(.param .u64 p) { .reg .f32 %f<4>; ret; }\n`;
    const ptx  = featuresFromPtx(ptxMod, "_Z8gemmHMMAK");
    const sass = featuresFromSass(lines.join("\n"), "_Z8gemmHMMAK");
    const out  = analyzePattern(ptx, sass);
    expect(out.uses_tensor_cores).toBe(true);
    expect(out.missing_tensor_cores).toBe(false);
    expect(out.missing_tensor_cores).toBe(false);
  });

  it("missing_tensor_cores_false_for_memory_bound_kernel", () => {
    // A memory-bound kernel (low compute_to_memory ratio) should NOT flag
    // missing_tensor_cores because tensor cores don't help bandwidth-limited code.
    let a = 0x0500;
    const lines = ["Function : _Z7memBndK", ""];
    for (let i = 0; i < 20; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R${i % 8}, [R${(i % 8) + 10}];`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} FFMA.RN R0, R1, R2, R3;`);  // one FFMA — low C/M ratio
    const ptxMod =
      PTX_HEAD + `\n.visible .entry _Z7memBndK(.param .u64 p) { .reg .f32 %f<4>; ret; }\n`;
    const ptx  = featuresFromPtx(ptxMod, "_Z7memBndK");
    const sass = featuresFromSass(lines.join("\n"), "_Z7memBndK");
    const out  = analyzePattern(ptx, sass);
    // compute_to_memory is very low, so threshold not met
    expect(out.missing_tensor_cores).toBe(false);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// New pattern signals: atomic_contention_risk / sfu_heavy / vectorization_score
//                      over_synchronized / fp16_scalar_risk / read_modify_write
// ─────────────────────────────────────────────────────────────────────────────

describe("pattern_model — inefficiency signals v2", () => {
  // Helper: minimal valid PTX module with a trivial kernel body.
  function trivialPtx(name: string): string {
    return (
      PTX_HEAD +
      `\n.visible .entry ${name}(.param .u64 p) {\n  .reg .f32 %f<4>; .reg .u64 %rd<4>;\n  ret;\n}\n`
    );
  }

  // ── atomic_contention_risk ────────────────────────────────────────────────
  //
  // Fires when global-memory atomics (ATOM/RED only — NOT ATOMS) exceed 5 % of
  // all global memory ops.  ATOMS targets shared memory (bank contention only,
  // not L2), so it does NOT contribute to this signal.
  // Condition: globalAtomicOps > 0 && globalOps > 0 && globalAtomicOps/globalOps > 0.05

  it("atomic_contention_risk_true_when_ratio_exceeds_5pct", () => {
    // 10 ATOM instructions out of ~11 global-ops ≈ 91 % → clearly above 5 %.
    let a = 0x1000;
    const lines = ["Function : _Z5atomCK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10; // 1 load
    for (let i = 0; i < 10; i++) {
      lines.push(`${sassHx(a)} ATOM.E.ADD [R4], R6;`);
      a += 0x10;
    }
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z5atomCK"), "_Z5atomCK");
    const sass = featuresFromSass(lines.join("\n"), "_Z5atomCK");
    const out  = analyzePattern(ptx, sass);
    expect(out.atomic_contention_risk).toBe(true);
    expect(out.atomic_contention_risk).toBe(true);
  });

  it("atomic_contention_risk_false_when_no_atomics", () => {
    // Kernel with only plain loads/stores — no atomics at all.
    let a = 0x1100;
    const lines = ["Function : _Z6noatomK", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R${i}, [R${i + 8}];`);
      a += 0x10;
    }
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z6noatomK"), "_Z6noatomK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6noatomK");
    const out  = analyzePattern(ptx, sass);
    expect(out.atomic_contention_risk).toBe(false);
  });

  it("atomic_contention_risk_false_when_only_shared_atomics", () => {
    // ATOMS (shared-memory atomics) must NOT trigger L2 contention risk.
    // A histogram kernel that accumulates into shared memory with ATOMS and
    // then does a single STG write should not be flagged.
    let a = 0x1180;
    const lines = ["Function : _Z9sharedAtK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10; // 1 global load
    for (let i = 0; i < 20; i++) {
      lines.push(`${sassHx(a)} ATOMS.ADD [R4], R6;`); a += 0x10; // shared atomics only
    }
    lines.push(`${sassHx(a)} STG.E.32 [R8], R0;`);  a += 0x10; // 1 global store
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`); a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z9sharedAtK"), "_Z9sharedAtK");
    const sass = featuresFromSass(lines.join("\n"), "_Z9sharedAtK");
    const out  = analyzePattern(ptx, sass);
    // global_atomic_ops = 0 → ratio = 0 → no L2 contention risk
    expect(out.atomic_contention_risk).toBe(false);
  });

  // ── sfu_heavy ─────────────────────────────────────────────────────────────
  //
  // Fires when > 8 MUFU instructions AND MUFU / (computeOps + MUFU + 1) > 0.15.

  it("sfu_heavy_true_when_mufu_exceeds_threshold", () => {
    // 12 MUFU + 4 FFMA → MUFU/(4+12+1)=12/17≈0.71 >> 0.15 and count > 8
    let a = 0x1200;
    const lines = ["Function : _Z4sfuHK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    for (let i = 0; i < 12; i++) {
      lines.push(`${sassHx(a)} MUFU.RCP R${i % 8}, R${(i % 8) + 1};`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z4sfuHK"), "_Z4sfuHK");
    const sass = featuresFromSass(lines.join("\n"), "_Z4sfuHK");
    const out  = analyzePattern(ptx, sass);
    expect(out.sfu_heavy).toBe(true);
    expect(out.sfu_heavy).toBe(true);
  });

  it("sfu_heavy_false_when_no_mufu", () => {
    // Pure FP32 kernel — zero MUFU instructions.
    let a = 0x1300;
    const lines = ["Function : _Z5nosfuK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    for (let i = 0; i < 20; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z5nosfuK"), "_Z5nosfuK");
    const sass = featuresFromSass(lines.join("\n"), "_Z5nosfuK");
    const out  = analyzePattern(ptx, sass);
    expect(out.sfu_heavy).toBe(false);
  });

  // ── vectorization_score ───────────────────────────────────────────────────
  //
  // Continuous score in [0.25, 1.0]:  1.0 = all LDG.128, 0.25 = all LDG.32.
  // Returns 0 when fewer than 4 typed loads are present (not enough data).

  it("vectorization_score_1_when_all_ldg128", () => {
    // 8 LDG.128 → weightedLanes = 8×4 = 32, totalLanes = 8×4 = 32 → score = 1.0
    let a = 0x1400;
    const lines = ["Function : _Z4vec1K", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDG.E.128 R${i * 4}, [R${i * 4 + 16}];`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
    const ptx  = featuresFromPtx(trivialPtx("_Z4vec1K"), "_Z4vec1K");
    const sass = featuresFromSass(lines.join("\n"), "_Z4vec1K");
    const out  = analyzePattern(ptx, sass);
    expect(out.vectorization_score).toBeCloseTo(1.0, 5);
    expect(out.vectorization_score).toBeCloseTo(1.0, 5);
  });

  it("vectorization_score_025_when_all_ldg32", () => {
    // 8 LDG.32 → weightedLanes = 8×1 = 8, totalLanes = 8×4 = 32 → score = 0.25
    let a = 0x1500;
    const lines = ["Function : _Z4vec4K", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R${i}, [R${i + 8}];`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
    const ptx  = featuresFromPtx(trivialPtx("_Z4vec4K"), "_Z4vec4K");
    const sass = featuresFromSass(lines.join("\n"), "_Z4vec4K");
    const out  = analyzePattern(ptx, sass);
    expect(out.vectorization_score).toBeCloseTo(0.25, 5);
  });

  it("vectorization_score_0_when_fewer_than_4_typed_loads", () => {
    // Only 2 typed LDG instructions → not enough data; score must be 0.
    let a = 0x1600;
    const lines = ["Function : _Z5fewLdK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    lines.push(`${sassHx(a)} LDG.E.128 R4, [R8];`); a += 0x10;
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
    const ptx  = featuresFromPtx(trivialPtx("_Z5fewLdK"), "_Z5fewLdK");
    const sass = featuresFromSass(lines.join("\n"), "_Z5fewLdK");
    const out  = analyzePattern(ptx, sass);
    expect(out.vectorization_score).toBe(0);
  });

  // ── over_synchronized ─────────────────────────────────────────────────────
  //
  // Fires when: loops > 0 && barriers >= 2 && barriers/loops > 1.5
  // (Requires PTX loop count from ptxFeatures.loops and SASS barrier count.)

  it("over_synchronized_true_when_barriers_exceed_loop_count", () => {
    // ptxOneLoop → loops = 1; SASS has 3 BAR.SYNC → ratio = 3 > 1.5.
    const ptxLooped =
      PTX_HEAD +
      `
.visible .entry _Z4ovrSK(
  .param .u64 p
)
{
  .reg .pred %p<2>;
  .reg .u32 %r<4>;
  .reg .u64 %rd<4>;

  mov.u32 %r0, 0;
  mov.u32 %r1, 4;

$L_loop:
  add.u32 %r0, %r0, 1;
  setp.lt.u32 %p0, %r0, %r1;
  @%p0 bra $L_loop;

  ret;
}
`;
    let a = 0x1700;
    const lines = ["Function : _Z4ovrSK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    lines.push(`${sassHx(a)} STG.E.32 [R4], R6;`);  a += 0x10;
    for (let i = 0; i < 3; i++) {
      lines.push(`${sassHx(a)} BAR.SYNC 0;`);
      a += 0x10;
    }
    for (let i = 0; i < 6; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(ptxLooped, "_Z4ovrSK");
    const sass = featuresFromSass(lines.join("\n"), "_Z4ovrSK");
    const out  = analyzePattern(ptx, sass);
    expect(out.over_synchronized).toBe(true);
    expect(out.over_synchronized).toBe(true);
  });

  it("over_synchronized_false_when_barrier_ratio_low", () => {
    // 1 BAR.SYNC with 2 loops → ratio = 0.5 < 1.5; should not fire.
    const ptxTwoLoops =
      PTX_HEAD +
      `
.visible .entry _Z5lowSyK(
  .param .u64 p
)
{
  .reg .pred %p<2>;
  .reg .u32 %r<4>;
  .reg .u64 %rd<4>;

  mov.u32 %r0, 0;
  mov.u32 %r1, 2;

$L_a:
  add.u32 %r0, %r0, 1;
  setp.lt.u32 %p0, %r0, %r1;
  @%p0 bra $L_a;

$L_b:
  add.u32 %r0, %r0, 1;
  setp.lt.u32 %p0, %r0, %r1;
  @%p0 bra $L_b;

  ret;
}
`;
    let a = 0x1800;
    const lines = ["Function : _Z5lowSyK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    lines.push(`${sassHx(a)} STG.E.32 [R4], R6;`);  a += 0x10;
    lines.push(`${sassHx(a)} BAR.SYNC 0;`);          a += 0x10; // only 1 barrier, 2 loops
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(ptxTwoLoops, "_Z5lowSyK");
    const sass = featuresFromSass(lines.join("\n"), "_Z5lowSyK");
    const out  = analyzePattern(ptx, sass);
    expect(out.over_synchronized).toBe(false);
  });

  it("over_synchronized_false_when_no_loops_in_ptx", () => {
    // Without at least one loop in PTX the signal is always false (no denominator).
    const ptxNoLoop =
      PTX_HEAD +
      `
.visible .entry _Z6noLoopK(
  .param .u64 p
)
{
  .reg .f32 %f<4>;
  .reg .u64 %rd<4>;
  ld.global.f32 %f0, [%rd1];
  ret;
}
`;
    let a = 0x1900;
    const lines = ["Function : _Z6noLoopK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    lines.push(`${sassHx(a)} STG.E.32 [R4], R6;`);  a += 0x10;
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} BAR.SYNC 0;`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(ptxNoLoop, "_Z6noLoopK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6noLoopK");
    const out  = analyzePattern(ptx, sass);
    expect(out.over_synchronized).toBe(false);
  });

  // ── fp16_scalar_risk ──────────────────────────────────────────────────────
  //
  // Fires when: fp16ArithOps > 8 && wmma_ops === 0 && fp16/computeOps > 0.2

  it("fp16_scalar_risk_true_for_heavy_hfma_no_hmma", () => {
    // 12 HFMA (scalar FP16) + 2 LDG → no HMMA tensor-core instructions.
    // fp16 ratio = 12/(12+1) ≈ 0.92 >> 0.2; wmma_ops === 0 → should fire.
    let a = 0x1a00;
    const lines = ["Function : _Z5fp16RK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    lines.push(`${sassHx(a)} LDG.E.32 R1, [R3];`);  a += 0x10;
    for (let i = 0; i < 12; i++) {
      lines.push(`${sassHx(a)} HFMA.F16 R${i % 8}, R${(i+1)%8}, R${(i+2)%8}, R${(i+3)%8};`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z5fp16RK"), "_Z5fp16RK");
    const sass = featuresFromSass(lines.join("\n"), "_Z5fp16RK");
    const out  = analyzePattern(ptx, sass);
    expect(out.fp16_scalar_risk).toBe(true);
    expect(out.fp16_scalar_risk).toBe(true);
  });

  it("fp16_scalar_risk_false_when_hmma_present", () => {
    // Same kernel shape but includes HMMA → wmma_ops > 0 so risk is false.
    let a = 0x1b00;
    const lines = ["Function : _Z6hmmaOKK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    for (let i = 0; i < 12; i++) {
      lines.push(`${sassHx(a)} HFMA.F16 R${i % 8}, R${(i+1)%8}, R${(i+2)%8}, R${(i+3)%8};`);
      a += 0x10;
    }
    // One HMMA instruction makes wmma_ops = 1 → suppresses fp16_scalar_risk.
    lines.push(`${sassHx(a)} HMMA.16816.F32 {R0,R1,R2,R3},{R4,R5},{R6,R7},{R0,R1,R2,R3};`);
    const ptx  = featuresFromPtx(trivialPtx("_Z6hmmaOKK"), "_Z6hmmaOKK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6hmmaOKK");
    const out  = analyzePattern(ptx, sass);
    expect(out.fp16_scalar_risk).toBe(false);
  });

  // ── read_modify_write ─────────────────────────────────────────────────────
  //
  // Fires when: storeToLoadRatio ∈ (0.5, 2.0) && computeToMemory < 1.0 && globalOps > 4

  it("read_modify_write_true_when_stores_approx_loads_low_compute", () => {
    // 6 loads + 5 stores (ratio ≈ 0.83, in range) + 3 FFMA (low compute/mem).
    // globalOps = 11 > 4; computeToMemory = 3/11 ≈ 0.27 < 1.0 → fires.
    let a = 0x1c00;
    const lines = ["Function : _Z3rmwK", ""];
    for (let i = 0; i < 6; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R${i}, [R${i + 8}];`);
      a += 0x10;
    }
    for (let i = 0; i < 5; i++) {
      lines.push(`${sassHx(a)} STG.E.32 [R${i + 8}], R${i};`);
      a += 0x10;
    }
    for (let i = 0; i < 3; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z3rmwK"), "_Z3rmwK");
    const sass = featuresFromSass(lines.join("\n"), "_Z3rmwK");
    const out  = analyzePattern(ptx, sass);
    expect(out.read_modify_write).toBe(true);
    expect(out.read_modify_write).toBe(true);
  });

  it("read_modify_write_false_when_compute_heavy", () => {
    // 2 loads + 1 store + 30 FFMA → computeToMemory = 30/3 = 10 >> 1.0 → no flag.
    let a = 0x1d00;
    const lines = ["Function : _Z5compHK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    lines.push(`${sassHx(a)} LDG.E.32 R1, [R3];`);  a += 0x10;
    lines.push(`${sassHx(a)} STG.E.32 [R4], R5;`);  a += 0x10;
    for (let i = 0; i < 30; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z5compHK"), "_Z5compHK");
    const sass = featuresFromSass(lines.join("\n"), "_Z5compHK");
    const out  = analyzePattern(ptx, sass);
    expect(out.read_modify_write).toBe(false);
  });

  it("read_modify_write_false_when_stores_much_less_than_loads", () => {
    // 10 loads + 1 store → storeToLoadRatio = 1/11 ≈ 0.09 < 0.5 → no flag.
    let a = 0x1e00;
    const lines = ["Function : _Z6strLowK", ""];
    for (let i = 0; i < 10; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R${i}, [R${i + 8}];`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} STG.E.32 [R8], R0;`);  a += 0x10;
    for (let i = 0; i < 3; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z6strLowK"), "_Z6strLowK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6strLowK");
    const out  = analyzePattern(ptx, sass);
    expect(out.read_modify_write).toBe(false);
  });
});

// ── Groups E / F / H ─────────────────────────────────────────────────────────

describe("pattern_model — stall inference, warp primitives, archetypes", () => {
  function trivialPtx(name: string): string {
    return (
      PTX_HEAD +
      `\n.visible .entry ${name}(.param .u64 p) {\n  .reg .f32 %f<4>; .reg .u64 %rd<4>;\n  ret;\n}\n`
    );
  }

  // ── E1: stall_memory_dependency ───────────────────────────────────────────
  //
  // Condition: stream_max_consecutive_loads > 4 AND stream_interleave_score < 0.2
  // (many back-to-back loads with almost no intervening compute)

  it("stall_memory_dependency_true_for_long_load_chain", () => {
    // 8 consecutive LDG instructions → maxLoadRun = 8 > 4.
    // No compute interspersed → interleave_score ≈ 0 < 0.2.
    let a = 0x2000;
    const lines = ["Function : _Z6ldChainK", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R${i}, [R${i + 8}];`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z6ldChainK"), "_Z6ldChainK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6ldChainK");
    const out  = analyzePattern(ptx, sass);
    expect(out.stall_memory_dependency).toBe(true);
    expect(out.stall_memory_dependency).toBe(true);
  });

  it("stall_memory_dependency_false_when_loads_interleaved", () => {
    // LDG–FFMA alternation → maxLoadRun = 1; interleave_score = 1.0.
    let a = 0x2100;
    const lines = ["Function : _Z8ldInterLK", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R${i}, [R${i + 8}];`);
      a += 0x10;
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z8ldInterLK"), "_Z8ldInterLK");
    const sass = featuresFromSass(lines.join("\n"), "_Z8ldInterLK");
    const out  = analyzePattern(ptx, sass);
    expect(out.stall_memory_dependency).toBe(false);
  });

  // ── E2: stall_memory_throttle ─────────────────────────────────────────────
  //
  // Condition: globalOps > 16 AND stream_interleave_score > 0.4 AND computeToMemory < 2.0

  it("stall_memory_throttle_true_for_high_mem_interleaved_low_compute", () => {
    // 20 LDG + 20 FFMA alternating → globalOps = 20 > 16, interleave = 1.0 > 0.4.
    // 20 compute / 20 mem = 1.0 < 2.0 → fires.
    let a = 0x2200;
    const lines = ["Function : _Z8memThrotK", ""];
    for (let i = 0; i < 20; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R${i % 16}, [R${(i % 8) + 8}];`);
      a += 0x10;
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z8memThrotK"), "_Z8memThrotK");
    const sass = featuresFromSass(lines.join("\n"), "_Z8memThrotK");
    const out  = analyzePattern(ptx, sass);
    expect(out.stall_memory_throttle).toBe(true);
  });

  // ── E3: stall_local_memory ────────────────────────────────────────────────
  //
  // Condition: (local_loads + local_stores) / total_instructions > 0.03

  it("stall_local_memory_true_for_heavy_spill_io", () => {
    // 5 LDL + 5 STL out of ~12 total = ~83 % >> 3 %.
    let a = 0x2300;
    const lines = ["Function : _Z6spillHK", ""];
    for (let i = 0; i < 5; i++) {
      lines.push(`${sassHx(a)} LDL R${i}, [R14+${i * 4}];`);
      a += 0x10;
      lines.push(`${sassHx(a)} STL [R14+${i * 4}], R${i};`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
    a += 0x10;
    lines.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
    a += 0x10;
    const ptx  = featuresFromPtx(trivialPtx("_Z6spillHK"), "_Z6spillHK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6spillHK");
    const out  = analyzePattern(ptx, sass);
    expect(out.stall_local_memory).toBe(true);
    expect(out.stall_local_memory).toBe(true);
  });

  it("stall_local_memory_false_when_no_spill", () => {
    let a = 0x2400;
    const lines = ["Function : _Z7noSpillK", ""];
    for (let i = 0; i < 10; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} LDG.E.32 R4, [R2];`);
    a += 0x10;
    const ptx  = featuresFromPtx(trivialPtx("_Z7noSpillK"), "_Z7noSpillK");
    const sass = featuresFromSass(lines.join("\n"), "_Z7noSpillK");
    const out  = analyzePattern(ptx, sass);
    expect(out.stall_local_memory).toBe(false);
  });

  // ── E4: stall_sync ────────────────────────────────────────────────────────
  //
  // Condition: barriers >= 2 AND workPerBarrier < 30 AND hasShared

  it("stall_sync_true_for_many_barriers_little_work", () => {
    // 4 barriers + some shared memory; workPerBarrier = (4 ldg + 2 ffma) / 4 = 1.5 < 30.
    let a = 0x2500;
    const lines = ["Function : _Z7syncHeavK", ""];
    lines.push(`${sassHx(a)} LDS.32 R0, [R2];`);  a += 0x10;
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);  a += 0x10;
    lines.push(`${sassHx(a)} BAR.SYNC 0;`);  a += 0x10;
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);  a += 0x10;
    lines.push(`${sassHx(a)} BAR.SYNC 0;`);  a += 0x10;
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);  a += 0x10;
    lines.push(`${sassHx(a)} BAR.SYNC 0;`);  a += 0x10;
    const ptx  = featuresFromPtx(trivialPtx("_Z7syncHeavK"), "_Z7syncHeavK");
    const sass = featuresFromSass(lines.join("\n"), "_Z7syncHeavK");
    const out  = analyzePattern(ptx, sass);
    expect(out.stall_sync).toBe(true);
    expect(out.stall_sync).toBe(true);
  });

  it("stall_sync_false_when_work_per_barrier_is_high", () => {
    // workPerBarrier = computeOps / (barriers + 1).
    // With 90 FFMAs and 2 barriers: 90 / (2+1) = 30.0 which is NOT < 30 → false.
    let a = 0x2600;
    const lines = ["Function : _Z8syncLightK", ""];
    lines.push(`${sassHx(a)} LDS.32 R0, [R2];`);  a += 0x10;
    for (let i = 0; i < 90; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} BAR.SYNC 0;`);  a += 0x10;
    lines.push(`${sassHx(a)} BAR.SYNC 0;`);  a += 0x10;
    const ptx  = featuresFromPtx(trivialPtx("_Z8syncLightK"), "_Z8syncLightK");
    const sass = featuresFromSass(lines.join("\n"), "_Z8syncLightK");
    const out  = analyzePattern(ptx, sass);
    expect(out.stall_sync).toBe(false);
  });

  // ── F2: uses_warp_shuffle ─────────────────────────────────────────────────

  it("uses_warp_shuffle_true_for_shfl_instructions", () => {
    let a = 0x2700;
    const lines = ["Function : _Z6wShflK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    lines.push(`${sassHx(a)} SHFL.SYNC.IDX R1, R0, R3, 0x1f;`);  a += 0x10;
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptx  = featuresFromPtx(trivialPtx("_Z6wShflK"), "_Z6wShflK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6wShflK");
    const out  = analyzePattern(ptx, sass);
    expect(out.uses_warp_shuffle).toBe(true);
    expect(out.uses_warp_shuffle).toBe(true);
  });

  it("uses_warp_shuffle_false_when_no_shfl", () => {
    let a = 0x2800;
    const lines = ["Function : _Z7noShflK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);  a += 0x10;
    const ptx  = featuresFromPtx(trivialPtx("_Z7noShflK"), "_Z7noShflK");
    const sass = featuresFromSass(lines.join("\n"), "_Z7noShflK");
    const out  = analyzePattern(ptx, sass);
    expect(out.uses_warp_shuffle).toBe(false);
  });

  // ── F3: uses_warp_vote ────────────────────────────────────────────────────

  it("uses_warp_vote_true_for_vote_instructions", () => {
    let a = 0x2900;
    const lines = ["Function : _Z6wVoteK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    lines.push(`${sassHx(a)} VOTE.SYNC.ALL P0, P1, 0xff;`);  a += 0x10;
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);  a += 0x10;
    const ptx  = featuresFromPtx(trivialPtx("_Z6wVoteK"), "_Z6wVoteK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6wVoteK");
    const out  = analyzePattern(ptx, sass);
    expect(out.uses_warp_vote).toBe(true);
    expect(out.uses_warp_vote).toBe(true);
  });

  // ── F4: warp_reduction_pattern ────────────────────────────────────────────
  //
  // Condition: warpShuffleOps > 0 AND sfuOps === 0 AND barriers > 0

  it("warp_reduction_pattern_true_for_shfl_barrier_no_sfu", () => {
    // SHFL + BAR.SYNC + FFMA (no MUFU) → classic warp reduction.
    let a = 0x2a00;
    const lines = ["Function : _Z6wRedK", ""];
    lines.push(`${sassHx(a)} LDS.32 R0, [R2];`);  a += 0x10;
    lines.push(`${sassHx(a)} SHFL.SYNC.DOWN R1, R0, 0x10, 0x1f;`);  a += 0x10;
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);  a += 0x10;
    lines.push(`${sassHx(a)} BAR.SYNC 0;`);  a += 0x10;
    const ptx  = featuresFromPtx(trivialPtx("_Z6wRedK"), "_Z6wRedK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6wRedK");
    const out  = analyzePattern(ptx, sass);
    expect(out.warp_reduction_pattern).toBe(true);
    expect(out.warp_reduction_pattern).toBe(true);
  });

  it("warp_reduction_pattern_false_when_sfu_present", () => {
    // SHFL + MUFU (sfu_ops > 0) → transcendental-heavy, not a pure reduction.
    let a = 0x2b00;
    const lines = ["Function : _Z6wNoRedK", ""];
    lines.push(`${sassHx(a)} SHFL.SYNC.DOWN R1, R0, 0x10, 0x1f;`);  a += 0x10;
    lines.push(`${sassHx(a)} MUFU.RCP R3, R4;`);  a += 0x10;
    lines.push(`${sassHx(a)} BAR.SYNC 0;`);  a += 0x10;
    const ptx  = featuresFromPtx(trivialPtx("_Z6wNoRedK"), "_Z6wNoRedK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6wNoRedK");
    const out  = analyzePattern(ptx, sass);
    expect(out.warp_reduction_pattern).toBe(false);
  });

  // ── H5: archetype = "activation" ─────────────────────────────────────────
  //
  // Condition: pattern === "elementwise" AND sfuOps > 0 AND !usesTensor
  // Build: SFU ops (MUFU) + a few loads/stores, no shared, no barriers, no HMMA → elementwise.

  it("archetype_activation_for_elementwise_with_sfu", () => {
    let a = 0x2c00;
    const lines = ["Function : _Z6actKernK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);  a += 0x10;
    for (let i = 0; i < 6; i++) {
      lines.push(`${sassHx(a)} MUFU.EX2 R${i % 8}, R${(i + 1) % 8};`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);  a += 0x10;
    lines.push(`${sassHx(a)} STG.E.32 [R4], R0;`);  a += 0x10;
    const ptx  = featuresFromPtx(trivialPtx("_Z6actKernK"), "_Z6actKernK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6actKernK");
    const out  = analyzePattern(ptx, sass);
    expect(out.archetype).toBe("activation");
  });

  // ── H3: archetype = "reduction" ──────────────────────────────────────────
  //
  // Condition: pattern === "reduction" AND barriers >= 2 AND sharedOps > 0
  //            AND computeToMemory < 4.0 AND !usesTensor

  it("archetype_reduction_for_reduction_pattern_with_shared", () => {
    // Reduction classification requires hasShared + hasBarrier + hasLoops.
    // Use ptxOneLoop() so loops >= 1 is detected from PTX.
    // SASS: many LDG+STG, shared loads, 2 barriers — satisfies reduction conditions.
    let a = 0x2d00;
    const lines = ["Function : _Z5k_loop", ""];
    for (let i = 0; i < 14; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R${i % 8}, [R${(i % 4) + 8}];`);
      a += 0x10;
    }
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} STG.E.32 [R8], R${i % 8};`);
      a += 0x10;
    }
    for (let i = 0; i < 12; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    for (let i = 0; i < 10; i++) {
      lines.push(`${sassHx(a)} LDS.128 R${(i % 4) * 4}, [R4];`);
      a += 0x10;
    }
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} STS.128 [R4], R${(i % 4) * 4};`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} BAR.SYNC 0;`);  a += 0x10;
    lines.push(`${sassHx(a)} BAR.SYNC 0;`);  a += 0x10;
    // ptxOneLoop supplies loops=1 which satisfies hasLoops for reduction classification.
    const ptx  = featuresFromPtx(ptxOneLoop(), "_Z5k_loop");
    const sass = featuresFromSass(lines.join("\n"), "_Z5k_loop");
    const out  = analyzePattern(ptx, sass);
    // class = "reduction"; then archetype resolves: reduction OR softmax.
    // No sfuOps so the softmax branch requires sfuOps > 4 → won't fire.
    // The reduction branch fires: barriers >= 2, sharedOps > 0, computeToMemory < 4.
    expect(out.class).toBe("reduction");
    expect(out.archetype).toBe("reduction");
  });
});

describe("pattern_model — group A/C derived signals", () => {
  function trivialPtx(name: string): string {
    return (
      PTX_HEAD +
      `\n.visible .entry ${name}(.param .u64 p) {\n  .reg .f32 %f<4>; .reg .u64 %rd<4>;\n  ret;\n}\n`
    );
  }

  it("computes_spill_and_productive_fractions", () => {
    let a = 0x2e00;
    const lines = ["Function : _Z8spillSigK", ""];
    lines.push(`${sassHx(a)} LDL R0, [R2];`); a += 0x10;
    lines.push(`${sassHx(a)} STL [R4], R0;`); a += 0x10;
    lines.push(`${sassHx(a)} LDG.E.32 R1, [R2];`); a += 0x10;
    lines.push(`${sassHx(a)} STG.E.32 [R4], R1;`); a += 0x10;
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`); a += 0x10;
    const out = analyzePattern(
      featuresFromPtx(trivialPtx("_Z8spillSigK"), "_Z8spillSigK"),
      featuresFromSass(lines.join("\n"), "_Z8spillSigK")
    );
    expect(out.spill_risk).toBe(true);
    expect(out.spill_severity).toBeGreaterThan(0.3);
    expect(out.productive_instruction_fraction).toBeGreaterThan(0.5);
  });

  it("computes_tensor_utilization_and_fp_to_int_ratio", () => {
    let a = 0x2f00;
    const lines = ["Function : _Z9tensorIntK", ""];
    for (let i = 0; i < 6; i++) {
      lines.push(`${sassHx(a)} HMMA.16816.F32 {R0,R1,R2,R3},{R4,R5},{R6,R7},{R0,R1,R2,R3};`);
      a += 0x10;
    }
    for (let i = 0; i < 2; i++) {
      lines.push(`${sassHx(a)} IADD R0, R1, R2;`);
      a += 0x10;
    }
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const out = analyzePattern(
      featuresFromPtx(trivialPtx("_Z9tensorIntK"), "_Z9tensorIntK"),
      featuresFromSass(lines.join("\n"), "_Z9tensorIntK")
    );
    expect(out.tensor_utilization_fraction).toBeGreaterThan(0.45);
    expect(out.fp_to_int_ratio).toBeGreaterThan(0.5);
  });

  it("computes_shared_reuse_and_store_uncoalesced_risk", () => {
    let a = 0x3000;
    const lines = ["Function : _Z10storeRiskK", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDS.32 R0, [R2];`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} BAR.SYNC 0;`); a += 0x10;
    lines.push(`${sassHx(a)} BAR.SYNC 0;`); a += 0x10;
    for (let i = 0; i < 6; i++) {
      lines.push(`${sassHx(a)} STG.E.32 [R4], R0;`);
      a += 0x10;
    }
    const out = analyzePattern(
      featuresFromPtx(ptxOneLoop(), "_Z5k_loop"),
      featuresFromSass(lines.join("\n"), "_Z10storeRiskK")
    );
    expect(out.shared_reuse_per_barrier).toBeGreaterThan(2);
    expect(out.store_uncoalesced_risk).toBe(true);
  });

  it("computes_warp_divergence_risk_from_loop_and_branch", () => {
    const ptx = featuresFromPtx(ptxTwoLoops(), "_Z8k_2loops");
    let a = 0x3100;
    const lines = ["Function : _Z8k_2loops", ""];
    lines.push(`${sassHx(a)} BRA 0x200;`); a += 0x10;
    lines.push(`${sassHx(a)} BRA 0x220;`); a += 0x10;
    lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`); a += 0x10;
    const out = analyzePattern(ptx, featuresFromSass(lines.join("\n"), "_Z8k_2loops"));
    expect(out.warp_divergence_risk).toBe(true);
  });

  // ── B5 regression: SASS zero-count must not fall back to PTX ─────────────────
  //
  // The old rule was `sassCount > 0 ? sass : ptx`.  When SASS was present but
  // reported 0 global ops (e.g. a pure shared-memory kernel), the code would
  // substitute a stale PTX value, corrupting globalOps, computeOps, barriers,
  // and branches.
  //
  // The fix uses `sassFeatures !== undefined` as the switch so that a legitimate
  // zero SASS count is preserved as 0, not replaced with a PTX non-zero value.
  it("B5 – SASS zero globalOps overrides stale PTX non-zero globalOps", () => {
    // Build a PTX with non-zero global loads (stale / leftover from a different kernel)
    const ptxWithGlobalLoads =
      PTX_HEAD +
      `
.visible .entry _Z6sharedK(.param .u64 p)
{
  .reg .f32 %f<4>;
  .reg .u64 %rd<4>;
  // These stale global loads must NOT appear in the fused result when SASS is provided
  ld.global.f32 %f0, [%rd0];
  ld.global.f32 %f1, [%rd0];
  ld.global.f32 %f2, [%rd0];
  ld.global.f32 %f3, [%rd0];
  ld.global.f32 %f0, [%rd0];
  ret;
}
`;

    // SASS for the same kernel — genuinely no global memory ops (pure shared)
    let a = 0x9000;
    const lines = ["Function : _Z6sharedK", ""];
    // Only shared loads + compute, zero global loads or stores
    for (let i = 0; i < 16; i++) {
      lines.push(`${sassHx(a)} LDS.128 R${(i % 4) * 4}, [R24];`);
      a += 0x10;
    }
    for (let i = 0; i < 32; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} BAR.SYNC 0;`);

    const ptx  = featuresFromPtx(ptxWithGlobalLoads, "_Z6sharedK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6sharedK");

    const out = analyzePattern(ptx, sass);

    // SASS reported 0 global ops — that must win over PTX's 5.
    expect(out.global_ops).toBe(0);
    // Kernel is purely shared-memory, so sharedOps > 0 from SASS.
    expect(out.shared_ops).toBeGreaterThan(0);
    // With zero global ops, computeToMemory → large → expect compute_heavy or tiled
    // (not elementwise which requires low compute/memory, not memory_bound).
    expect(["compute_heavy", "tiled", "reduction", "irregular"]).toContain(out.class);
  });

  // ── B9: multi-kernel PTX sum must suppress density-based signals ─────────────
  //
  // When PTX features are summed across N kernels (kernelCount > 1), loop and
  // branch counts span kernel boundaries.  Density ratios like warp_divergence_risk
  // and over_synchronized become meaningless and must be forced to false.
  it("B9 – multi-kernel PTX sum suppresses warp_divergence_risk and over_synchronized", () => {
    // Simulate the result of sumPtxInstructionFeatures across two kernels:
    // each kernel has 2 loops and 2 barriers → sum has 4 loops and 4 barriers.
    // The ratio barriers/loops = 1.0 < 1.5 so over_synchronized wouldn't fire anyway,
    // but branches * loops / compute = (4 * 4) / (1 + 1) = 8 > 0.01 → warp_divergence_risk would fire.
    //
    // With kernelCount = 2 these must be suppressed.
    const sumFeatures: import("../src/analyzer/ptx_features").PtxInstructionFeatures = {
      global_loads: 8,
      global_stores: 4,
      fma: 0,
      add: 1,
      mul: 0,
      barrier: 4,
      reg_decl_lines: 4,
      branches: 4,
      loops: 4,
      kernelCount: 2,  // <-- multi-kernel sum
    };

    const out = analyzePattern(sumFeatures);

    // These density-based signals must be suppressed when kernelCount > 1 and no SASS
    expect(out.warp_divergence_risk).toBe(false);
    expect(out.over_synchronized).toBe(false);
    expect(out.high_looping).toBe(false);
    expect(out.complex_kernel).toBe(false);
    // The flag must be set
    expect(out.multi_kernel_ptx).toBe(true);
  });

  it("B9 – single-kernel PTX (kernelCount = 1) does NOT suppress signals", () => {
    // Same raw counts as above, but kernelCount = 1 (single kernel).
    // warp_divergence_risk and over_synchronized should fire as normal.
    const singleFeatures: import("../src/analyzer/ptx_features").PtxInstructionFeatures = {
      global_loads: 8,
      global_stores: 4,
      fma: 0,
      add: 1,
      mul: 0,
      barrier: 4,
      reg_decl_lines: 4,
      branches: 4,
      loops: 4,
      kernelCount: 1,  // single kernel
    };

    const out = analyzePattern(singleFeatures);

    // warp_divergence_risk should fire: (4 * 4) / (1 + 1) = 8 > 0.01
    expect(out.warp_divergence_risk).toBe(true);
    expect(out.multi_kernel_ptx).toBe(false);
  });

  it("B9 – SASS present overrides PTX kernelCount suppression", () => {
    // When SASS is available (hasSass = true), we use SASS counts, not PTX sums.
    // Suppression must NOT fire even if kernelCount > 1 (SASS data is reliable).
    const sumFeatures: import("../src/analyzer/ptx_features").PtxInstructionFeatures = {
      global_loads: 8,
      global_stores: 4,
      fma: 0,
      add: 1,
      mul: 0,
      barrier: 4,
      reg_decl_lines: 4,
      branches: 4,
      loops: 4,
      kernelCount: 3,  // multi-kernel sum, but SASS is present
    };

    let a = 0xd000;
    const lines = ["Function : _Z5sassK", ""];
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`); a += 0x10;
    }
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`); a += 0x10;
    }
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} BRA 0x200;`); a += 0x10;
    }

    const sass = featuresFromSass(lines.join("\n"), "_Z5sassK");
    const out = analyzePattern(sumFeatures, sass);

    // SASS overrides PTX: suppression must NOT apply (hasSass = true)
    expect(out.multi_kernel_ptx).toBe(false);
  });

  it("B9 – SASS zero barriers overrides stale PTX non-zero barriers", () => {
    // PTX with barriers
    const ptxWithBarriers =
      PTX_HEAD +
      `
.visible .entry _Z9noBarrierK(.param .u64 p)
{
  .reg .f32 %f<4>; .reg .u64 %rd<4>;
  ld.global.f32 %f0, [%rd0];
  bar.sync 0;   // PTX has a barrier
  bar.sync 0;   // ... and another
  add.f32 %f1, %f0, %f0;
  st.global.f32 [%rd0], %f1;
  ret;
}
`;

    // SASS for the same kernel — no BAR.SYNC emitted (compiler elided it)
    let a = 0xa000;
    const lines = ["Function : _Z9noBarrierK", ""];
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`);
      a += 0x10;
    }
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    for (let i = 0; i < 8; i++) {
      lines.push(`${sassHx(a)} STG.E.32 [R2], R0;`);
      a += 0x10;
    }
    // Explicitly no BAR.SYNC lines

    const ptx  = featuresFromPtx(ptxWithBarriers, "_Z9noBarrierK");
    const sass = featuresFromSass(lines.join("\n"), "_Z9noBarrierK");

    const out = analyzePattern(ptx, sass);

    // SASS reported 0 barriers — must win over PTX's 2.
    expect(out.barriers).toBe(0);
    // Without barriers (or shared), can't be tiled or reduction.
    expect(["elementwise", "compute_heavy", "control_heavy"]).toContain(out.class);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Gap 5 — tensor_utilization_fraction / productive_instruction_fraction must
//         be `undefined` (not 0) for PTX-only kernels.
// ─────────────────────────────────────────────────────────────────────────────

describe("pattern_model — Gap 5: PTX-only utilisation fields are undefined", () => {
  it("tensor_utilization_fraction_is_undefined_when_no_sass", () => {
    const ptxMod =
      PTX_HEAD +
      `
.visible .entry _Z6ptxOnK(.param .u64 p) {
  .reg .f32 %f<4>; .reg .u64 %rd<4>;
  ld.global.f32 %f0, [%rd1];
  fma.rn.f32 %f1, %f0, %f0, %f0;
  ret;
}
`;
    const out = analyzePattern(featuresFromPtx(ptxMod, "_Z6ptxOnK"));
    // Old behaviour reported 0 here; that masked "no SASS data" as
    // "verified zero tensor work".  Gap 5 fix makes both fields undefined.
    expect(out.tensor_utilization_fraction).toBeUndefined();
    expect(out.productive_instruction_fraction).toBeUndefined();
  });

  it("tensor_utilization_fraction_is_defined_when_sass_present", () => {
    let a = 0x4000;
    const lines = ["Function : _Z7ptxSasK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`); a += 0x10;
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} HMMA.16816.F32 {R0,R1,R2,R3},{R4,R5},{R6,R7},{R0,R1,R2,R3};`);
      a += 0x10;
    }
    for (let i = 0; i < 4; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    const ptxMod =
      PTX_HEAD +
      `\n.visible .entry _Z7ptxSasK(.param .u64 p) { .reg .f32 %f<4>; ret; }\n`;
    const ptx  = featuresFromPtx(ptxMod, "_Z7ptxSasK");
    const sass = featuresFromSass(lines.join("\n"), "_Z7ptxSasK");
    const out  = analyzePattern(ptx, sass);
    expect(out.tensor_utilization_fraction).not.toBeUndefined();
    expect(out.productive_instruction_fraction).not.toBeUndefined();
    // 4 HMMA out of (4 arithmetic + 4 tensor + 1) = 4/9 ≈ 0.44
    expect(out.tensor_utilization_fraction!).toBeGreaterThan(0.4);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Gap 8 — over_synchronized falls back to SASS back-edges when PTX `loops` = 0
//         (handles fully-unrolled outer loops the compiler removed in PTX).
// ─────────────────────────────────────────────────────────────────────────────

describe("pattern_model — Gap 8: over_synchronized via SASS back-edges", () => {
  function trivialPtxNoLoops(name: string): string {
    return (
      PTX_HEAD +
      `\n.visible .entry ${name}(.param .u64 p) {\n  .reg .f32 %f<4>; .reg .u64 %rd<4>;\n  ld.global.f32 %f0, [%rd1];\n  ret;\n}\n`
    );
  }

  it("backward_BRA_substitutes_for_PTX_loops_in_over_synchronized_check", () => {
    // PTX has zero loops (compiler unrolled it), but SASS still emits a
    // backward BRA per logical iteration AND multiple BAR.SYNCs inside the
    // unrolled body.  Gap 8 lets `over_synchronized` fire on this case.
    const lines: string[] = ["Function : _Z6unrlSK", ""];
    let a = 0x5000;
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`); a += 0x10;
    for (let i = 0; i < 3; i++) {
      lines.push(`${sassHx(a)} BAR.SYNC 0;`); a += 0x10;
    }
    for (let i = 0; i < 6; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`); a += 0x10;
    }
    // One backward BRA from 0x5100 → 0x5000 closes the loop.
    lines.push(`${sassHx(0x5100)} BRA 0x5000;`);
    const ptx  = featuresFromPtx(trivialPtxNoLoops("_Z6unrlSK"), "_Z6unrlSK");
    const sass = featuresFromSass(lines.join("\n"), "_Z6unrlSK");
    expect(ptx.loops).toBe(0);          // confirm PTX has no loops
    expect(sass.back_edges).toBe(1);    // confirm we counted the back-edge
    const out = analyzePattern(ptx, sass);
    // 3 barriers / 1 effective loop = 3 > 1.5 → over_synchronized fires.
    expect(out.over_synchronized).toBe(true);
  });

  it("no_back_edges_and_no_PTX_loops_keeps_over_synchronized_false", () => {
    // Sanity: forward-only BRA must not trigger the loop-equivalent fallback.
    let a = 0x5200;
    const lines = ["Function : _Z9noLoopBSK", ""];
    lines.push(`${sassHx(a)} LDG.E.32 R0, [R2];`); a += 0x10;
    for (let i = 0; i < 3; i++) {
      lines.push(`${sassHx(a)} BAR.SYNC 0;`); a += 0x10;
    }
    lines.push(`${sassHx(a)} BRA 0x5300;`); a += 0x10;     // forward branch
    lines.push(`${sassHx(0x5300)} FFMA.FTZ R0, R1, R2, R3;`);
    const ptx  = featuresFromPtx(trivialPtxNoLoops("_Z9noLoopBSK"), "_Z9noLoopBSK");
    const sass = featuresFromSass(lines.join("\n"), "_Z9noLoopBSK");
    expect(ptx.loops).toBe(0);
    expect(sass.back_edges).toBe(0);
    const out = analyzePattern(ptx, sass);
    expect(out.over_synchronized).toBe(false);
  });
});
