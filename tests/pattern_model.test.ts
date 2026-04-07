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
    expect(out.pattern_micro.streaming).toBe(true);
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
    expect(outInterleaved.pattern_micro.interleaving).toBe("interleaved");
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
    expect(outStacked.pattern_micro.interleaving).toBe("stacked");
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
    expect(out.pattern_micro.high_looping).toBe(true);
    expect(out.pattern_micro.sync_heavy).toBe(true);
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
    expect(out.pattern_micro.control_irregular).toBe(true);
    expect(out.pattern_micro.control_dominated).toBe(true);
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
    expect(out.pattern_micro.complex_kernel).toBe(true);
    expect(out.pattern_micro.tensor_dominated).toBe(true);
  });
});
