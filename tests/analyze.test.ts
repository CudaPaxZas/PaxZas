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

// ─────────────────────────────────────────────────────────────────────────────
// Gap 9 — analysis_mode "native" detection compares cc keys, not literal SMs
//
// `sm_87` (Jetson Orin) and `sm_86` (Ampere mobile) both map to compute
// capability 8.6 in SM_VERSION_TO_CC.  Before Gap 9 a SASS dump for `sm_87`
// against an Ampere-mobile preset was reported as `cross-arch-what-if`; after
// the fix both resolve to the same cc key and the analysis_mode is "native".
// ─────────────────────────────────────────────────────────────────────────────

describe("analyze() — Gap 9: analysis_mode native via cc-key", () => {
  const PTX_BARE_LDG = `
.version 7.4
.target sm_86
.address_size 64
.visible .entry _Z6smCcKv()
.maxntid 256, 1, 1
.maxnreg 48
{
  .reg .f32 %f<2>;
  .reg .u64 %rd<2>;
  ld.global.f32 %f1, [%rd1];
  ret;
}
`;

  // The "ampere-like-default" preset has smTag = sm_86 (cc 8.6); we use it as
  // the reference target for the cc-key normalisation tests below.

  it("matches sm_86 SASS against ampere-like-default preset as native", async () => {
    const sass = [
      "Fatbin elf code:",
      "code for sm_86",
      "Function : _Z6smCcKv",
      "    /*0010*/ LDG.E.32 R0, [R2];",
    ].join("\n");
    const r = await analyze(PTX_BARE_LDG, {}, undefined, sass, "ampere-like-default");
    expect(r.error).toBeUndefined();
    expect(r.analysis_mode).toBe("native");
    expect(r.sass_detected_targets).toContain("sm_86");
  });

  it("matches sm_87 (Jetson Orin) SASS against ampere-like-default (sm_86) preset as native", async () => {
    // Both sm_87 and sm_86 map to cc 8.6 in SM_VERSION_TO_CC.  Pre-Gap-9 the
    // literal-equality check would flag this as cross-arch-what-if; now the
    // cc-key normalisation should classify it as "native".
    const sass = [
      "Fatbin elf code:",
      "code for sm_87",
      "Function : _Z6smCcKv",
      "    /*0010*/ LDG.E.32 R0, [R2];",
    ].join("\n");
    const r = await analyze(PTX_BARE_LDG, {}, undefined, sass, "ampere-like-default");
    expect(r.error).toBeUndefined();
    expect(r.analysis_mode).toBe("native");
    expect(r.sass_detected_targets).toContain("sm_87");
  });

  it("flags sm_90 SASS against ampere-like-default (sm_86) preset as cross-arch-what-if", async () => {
    // Genuinely different architecture (Hopper vs Ampere) — must NOT be native.
    const sass = [
      "Fatbin elf code:",
      "code for sm_90",
      "Function : _Z6smCcKv",
      "    /*0010*/ LDG.E.32 R0, [R2];",
    ].join("\n");
    const r = await analyze(PTX_BARE_LDG, {}, undefined, sass, "ampere-like-default");
    expect(r.error).toBeUndefined();
    expect(r.analysis_mode).toBe("cross-arch-what-if");
  });

  it("uses preset-only-what-if when no SASS targets are detected", async () => {
    const r = await analyze(PTX_BARE_LDG, {}, undefined, undefined, "ampere-like-default");
    expect(r.error).toBeUndefined();
    expect(r.analysis_mode).toBe("preset-only-what-if");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// I8 / I3 — SASS-only path honours explicit launch hints (threads/shared/regs/grid)
//
// Pre-I8 the SASS-only branch silently used threads=256, shared=0, regs=32 (or
// the SASS-inferred regs) regardless of what the caller passed. Post-I8 the
// caller's launch dictionary is respected, the I3 grid hint is propagated to
// `analyzeKernel` via `gridBlocks`, and the `sass_note` text reflects which
// fields came from hints vs synthetic defaults.
// ─────────────────────────────────────────────────────────────────────────────

describe("analyze() — I8/I3 SASS-only respects launch hints", () => {
  const SASS_ONLY = [
    "Fatbin elf code:",
    "code for sm_86",
    "Function : _Z6sassOK",
    "    /*0010*/ LDG.E.32 R0, [R2];",
    "    /*0020*/ FFMA.FTZ R3, R0, R1, R2;",
    "    /*0030*/ STG.E [R8], R3;",
  ].join("\n");

  it("uses synthetic defaults when no launch hints are passed", async () => {
    const r = await analyze(SASS_ONLY, {}, undefined, undefined, "ampere-like-default");
    expect(r.error).toBeUndefined();
    expect(r.kind).toBe("sass_only");
    expect(r.sass_note).toContain("no launch hints");
    expect(r.sass_note).toContain("threads=256");
  });

  it("respects threads/shared/registers hints in SASS-only mode", async () => {
    const r = await analyze(
      SASS_ONLY,
      { threads: 128, shared: 4096, registers: 40 },
      undefined,
      undefined,
      "ampere-like-default"
    );
    expect(r.error).toBeUndefined();
    expect(r.sass_note).toContain("threads=128");
    expect(r.sass_note).toContain("shared=4096");
    expect(r.sass_note).toContain("registers=40");
    expect(r.sass_note).not.toContain("synthetic defaults");
  });

  it("propagates grid hint through SASS-only path to estimated_sm_utilization (I3)", async () => {
    const r = await analyze(
      SASS_ONLY,
      { threads: 128, grid: 6 },
      undefined,
      undefined,
      "a100"
    );
    expect(r.error).toBeUndefined();
    expect(r.sass_note).toContain("grid≤6 blocks");
    expect(r.kernel?.estimated_sm_utilization).toContain("grid≤6");
  });
});
