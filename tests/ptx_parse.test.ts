/**
 * Mirrors tests/test_ptx_parse.py
 */
import { describe, expect, it } from "vitest";
import {
  maxThreadsFromMaxntid,
  parsePtxKernelHints,
} from "../src/analyzer/ptx_parse";

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
.shared .align 4 .b8 smem[1024];
ret;
}
`;

const SAMPLE_PTX_DYNAMIC = `
.version 8.0
.target sm_80
.visible .entry _Z10dynsharedv(
.param .u64 _Z10dynsharedv_param_0
)
.maxntid 128, 1, 1
.maxnreg 32
{
.extern .shared .align 16 .b8 dyn_smem[];
ret;
}
`;

describe("ptx_parse (Python parity)", () => {
  it("parse_ptx_hints", () => {
    const h = parsePtxKernelHints(SAMPLE_PTX, undefined);
    expect(h.kernelName).toBeDefined();
    expect(h.maxnreg).toBe(48);
    expect(h.maxntid).toEqual([256, 1, 1]);
    expect(maxThreadsFromMaxntid(h.maxntid)).toBe(256);
    expect(h.staticSharedBytes).toBe(1024);
    expect(h.dynamicSharedDetected).toBe(false);
  });

  it("parse_ptx_hints_detects_dynamic_shared", () => {
    const h = parsePtxKernelHints(SAMPLE_PTX_DYNAMIC, undefined);
    expect(h.kernelName).toBe("_Z10dynsharedv");
    expect(h.staticSharedBytes).toBe(0);
    expect(h.dynamicSharedDetected).toBe(true);
  });
});
