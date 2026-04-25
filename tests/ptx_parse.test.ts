/**
 * Mirrors tests/test_ptx_parse.py
 */
import { describe, expect, it } from "vitest";
import {
  maxThreadsFromMaxntid,
  parsePtxKernelHints,
  listPtxEntryNames,
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

  // B2 regression: element count must be multiplied by the element byte-width.
  // Before the fix, all declarations were treated as raw bytes regardless of type,
  // so `.u32 buf[256]` wrongly reported 256 bytes instead of 1024.
  it("B2 – typed shared arrays use element size (u32×256 = 1024 bytes)", () => {
    const ptx = `
.version 8.0
.target sm_80
.visible .entry _Z6kernel(
.param .u64 p
)
{
.shared .align 4 .u32  tile[256];
.shared .align 8 .f64  dbuf[64];
.shared .align 2 .b16  hbuf[512];
.shared .align 4 .b8   raw[128];
ret;
}
`;
    const h = parsePtxKernelHints(ptx, undefined);
    // tile:  256 × 4 = 1024
    // dbuf:   64 × 8 =  512
    // hbuf:  512 × 2 = 1024
    // raw:   128 × 1 =  128
    // total: 2688 bytes
    expect(h.staticSharedBytes).toBe(2688);
  });

  it("B2 – .b8 (raw byte) declarations are counted as 1 byte per element", () => {
    // The existing test fixture already exercises this path (.b8 smem[1024])
    // but this makes the expectation explicit.
    const h = parsePtxKernelHints(SAMPLE_PTX, undefined);
    expect(h.staticSharedBytes).toBe(1024); // 1024 × 1 byte
  });
});
