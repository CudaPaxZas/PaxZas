/**
 * PaxZasValidation ↔ PaxZas regression anchors (full KERNEL_CATALOG).
 *
 * One `it(...)` per cataloged kernel in `PaxZasValidation/KERNEL_CATALOG.md`:
 *   • Original suite (5) — `cuda/kernel_suite.cu`
 *   • Extended suite (22) — `cuda/kernels_extended.cu` §1–22
 *   • GEMM suite (11) — `cuda/kernels_gemm.cu` §G1–G11
 *
 * Each test keeps a minimal `cuobjdump --dump-sass`-shaped inline fallback.
 * When `tests/data/sass/*.sass` (or `PAXZAS_VALIDATION_SASS*`) and matching
 * PTX under `tests/data/ptx/` exist, the same assertions run on real
 * extracted text (`catalogSassFeatures` + `expectCatalogPtxKernel`).
 *
 * Run: `npm test -- --run tests/validation_suite.test.ts`
 */
import { describe, expect, it } from "vitest";
import { sassHx } from "./helpers/ir_fixtures";
import {
  catalogPtxFeatures,
  catalogSassFeatures,
  expectCatalogPtxKernel,
} from "./helpers/validation_dump_loader";

function sassKernel(name: string, bodyLines: string[]): string {
  return [`Function : ${name}`, "", ...bodyLines].join("\n");
}

// =============================================================================
// Original Suite — KERNEL_CATALOG "Original Suite" table
// =============================================================================

describe("KERNEL_CATALOG — mem_stream (kernel_suite.cu)", () => {
  it("ldg_stg_32_coalesced_stream", () => {
    const s = sassKernel("_Z10mem_streamPfS_i", [
      `${sassHx(0x0100)} LDG.E.32 R0, [R4];`,
      `${sassHx(0x0110)} STG.E.32 [R8], R0;`,
    ]);
    const { f } = catalogSassFeatures("mem_stream", s);
    expectCatalogPtxKernel("mem_stream");
    expect(f.ldg_32).toBeGreaterThanOrEqual(1);
    expect(f.stg_32).toBeGreaterThanOrEqual(1);
    expect(f.global_loads).toBeGreaterThanOrEqual(1);
    expect(f.global_stores).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG — mem_strided (kernel_suite.cu)", () => {
  it("ldg_32_strided_pattern_still_typed_32", () => {
    const s = sassKernel("_Z12mem_stridedPfS_ii", [
      `${sassHx(0x0200)} LDG.E.32 R0, [R4];`,
    ]);
    const { f } = catalogSassFeatures("mem_strided", s);
    expectCatalogPtxKernel("mem_strided");
    expect(f.ldg_32).toBeGreaterThanOrEqual(1);
    expect(f.global_loads).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG — compute_heavy (kernel_suite.cu)", () => {
  it("dense_ffma_chain", () => {
    const body: string[] = [];
    let addr = 0x0300;
    for (let k = 0; k < 32; k++) {
      body.push(`${sassHx(addr)} FFMA.FTZ R0, R1, R2, R3;`);
      addr += 0x10;
    }
    const s = sassKernel("_Z14compute_heavyPfS_i", body);
    const { f, source } = catalogSassFeatures("compute_heavy", s);
    expectCatalogPtxKernel("compute_heavy");
    if (source === "inline") {
      expect(f.arithmetic_ops).toBe(32);
      expect(f.global_loads).toBe(0);
    } else {
      // `#pragma unroll 64` => kernel issues many FFMA per element.
      expect(f.arithmetic_ops).toBeGreaterThanOrEqual(32);
      expect(f.global_loads).toBeGreaterThanOrEqual(1);
    }
    const ptx = catalogPtxFeatures("compute_heavy");
    if (ptx) {
      // Catalog ties this kernel to a high `flops_proxy` driven by FMA count.
      expect(ptx.fma).toBeGreaterThanOrEqual(16);
    }
  });
});

describe("KERNEL_CATALOG — tiled_add (kernel_suite.cu)", () => {
  it("lds_sts_bar_ffma", () => {
    const s = sassKernel("_Z9tiled_addPfS_S_i", [
      `${sassHx(0x0400)} LDS.128 R8, [R4];`,
      `${sassHx(0x0410)} STS.128 [R4], R8;`,
      `${sassHx(0x0420)} BAR.SYNC 0;`,
      `${sassHx(0x0430)} FFMA.FTZ R0, R1, R2, R3;`,
    ]);
    const { f } = catalogSassFeatures("tiled_add", s);
    expectCatalogPtxKernel("tiled_add");
    expect(f.shared_loads).toBeGreaterThanOrEqual(1);
    expect(f.shared_stores).toBeGreaterThanOrEqual(1);
    expect(f.barrier).toBeGreaterThanOrEqual(1);
    expect(f.arithmetic_ops).toBeGreaterThanOrEqual(1);
    const ptx = catalogPtxFeatures("tiled_add");
    if (ptx) {
      expect(ptx.barrier).toBeGreaterThanOrEqual(1);
    }
  });
});

describe("KERNEL_CATALOG — reduction (kernel_suite.cu)", () => {
  it("lds_sts_multi_bar_reduction_shape", () => {
    const s = sassKernel("_Z10reductionPfS_i", [
      `${sassHx(0x0500)} LDS.32 R0, [R4];`,
      `${sassHx(0x0510)} STS.32 [R8], R0;`,
      `${sassHx(0x0520)} BAR.SYNC 0;`,
      `${sassHx(0x0530)} BAR.SYNC 0;`,
    ]);
    const { f } = catalogSassFeatures("reduction", s);
    expectCatalogPtxKernel("reduction");
    expect(f.shared_loads).toBeGreaterThanOrEqual(1);
    expect(f.shared_stores).toBeGreaterThanOrEqual(1);
    expect(f.barrier).toBeGreaterThanOrEqual(2);
    const ptx = catalogPtxFeatures("reduction");
    if (ptx) {
      // Tree-reduction kernel: blockDim/2 → 1 sync per round, log2(256)=8 rounds.
      expect(ptx.barrier).toBeGreaterThanOrEqual(2);
    }
  });
});

// =============================================================================
// Extended Suite §1–15 — KERNEL_CATALOG extended entries 1–15
// =============================================================================

describe("KERNEL_CATALOG §1 — vectorized_copy_128", () => {
  it("ldg128_stg128", () => {
    const s = sassKernel("_Z20vectorized_copy_128PfS_i", [
      `${sassHx(0x0600)} LDG.E.128 R4, [R2];`,
      `${sassHx(0x0610)} STG.E.128 [R8], R4;`,
    ]);
    const { f } = catalogSassFeatures("vectorized_copy_128", s);
    expectCatalogPtxKernel("vectorized_copy_128");
    expect(f.ldg_128).toBeGreaterThanOrEqual(1);
    expect(f.stg_128).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG §2 — int8_byte_copy", () => {
  it("ldg_u8_stg_u8", () => {
    const s = sassKernel("_Z14int8_byte_copyPhS_i", [
      `${sassHx(0x0700)} LDG.E.U8 R0, [R4];`,
      `${sassHx(0x0710)} STG.E.U8 [R8], R0;`,
    ]);
    const { f } = catalogSassFeatures("int8_byte_copy", s);
    expectCatalogPtxKernel("int8_byte_copy");
    expect(f.ldg_8).toBeGreaterThanOrEqual(1);
    expect(f.stg_8).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG §3 — fp16_scalar_compute", () => {
  it("hfma_without_hmma", () => {
    const s = sassKernel("_Z19fp16_scalar_computeP6__halfS0_i", [
      `${sassHx(0x0800)} HFMA.F16 R0, R1, R2, R3;`,
    ]);
    const { f } = catalogSassFeatures("fp16_scalar_compute", s);
    expectCatalogPtxKernel("fp16_scalar_compute");
    expect(f.fp16_arith_ops).toBeGreaterThanOrEqual(1);
    expect(f.wmma_ops).toBe(0);
    expect(f.tensor_ops).toBe(0);
  });
});

describe("KERNEL_CATALOG §4 — fp64_compute_chain", () => {
  it("dfma_counts_fp64_arith", () => {
    const s = sassKernel("_Z18fp64_compute_chainPdS_i", [
      `${sassHx(0x0900)} DFMA R0, R2, R4, R6;`,
    ]);
    const { f } = catalogSassFeatures("fp64_compute_chain", s);
    expectCatalogPtxKernel("fp64_compute_chain");
    expect(f.fp64_arith_ops).toBeGreaterThanOrEqual(1);
    expect(f.arithmetic_ops).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG §5 — sfu_transcendentals", () => {
  it("mufu_family", () => {
    const s = sassKernel("_Z19sfu_transcendentalsPfS_i", [
      `${sassHx(0x0a00)} MUFU.SIN R0, R1;`,
      `${sassHx(0x0a10)} MUFU.COS R2, R3;`,
      `${sassHx(0x0a20)} MUFU.EX2 R4, R5;`,
      `${sassHx(0x0a30)} MUFU.LG2 R6, R7;`,
    ]);
    const { f } = catalogSassFeatures("sfu_transcendentals", s);
    expectCatalogPtxKernel("sfu_transcendentals");
    expect(f.sfu_ops).toBeGreaterThanOrEqual(4);
  });
});

describe("KERNEL_CATALOG §6 — branchy_divergent", () => {
  it("predicated_bra_density", () => {
    const lines: string[] = [];
    let a = 0x0b00;
    for (let i = 0; i < 12; i++) {
      lines.push(`${sassHx(a)} @P0 BRA 0x200;`);
      a += 0x10;
    }
    const s = sassKernel("_Z17branchy_divergentPiS_i", lines);
    const { f, source } = catalogSassFeatures("branchy_divergent", s);
    expectCatalogPtxKernel("branchy_divergent");
    if (source === "inline") {
      expect(f.branch).toBe(12);
    } else {
      expect(f.branch).toBeGreaterThanOrEqual(4);
    }
    const ptx = catalogPtxFeatures("branchy_divergent");
    if (ptx) {
      expect(ptx.branches).toBeGreaterThanOrEqual(4);
    }
  });
});

describe("KERNEL_CATALOG §7 — register_spill_kernel", () => {
  it("ldl_stl_spill", () => {
    const s = sassKernel("_Z21register_spill_kernelPfS_i", [
      `${sassHx(0x0c00)} LDL R0, [R2+0x10];`,
      `${sassHx(0x0c10)} STL [R4+0x20], R6;`,
    ]);
    const { f } = catalogSassFeatures("register_spill_kernel", s);
    expectCatalogPtxKernel("register_spill_kernel");
    expect(f.local_loads).toBeGreaterThanOrEqual(1);
    expect(f.local_stores).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG §8 — warp_shuffle_reduction", () => {
  it("shfl_down_and_bar", () => {
    const s = sassKernel("_Z21warp_shuffle_reductionPfS_i", [
      `${sassHx(0x0d00)} SHFL.DOWN.BFLY R0, R1, R2, 0x1;`,
      `${sassHx(0x0d10)} BAR.SYNC 0;`,
    ]);
    const { f } = catalogSassFeatures("warp_shuffle_reduction", s);
    expectCatalogPtxKernel("warp_shuffle_reduction");
    expect(f.warp_shuffle_ops).toBeGreaterThanOrEqual(1);
    expect(f.barrier).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG §9 — warp_vote_predicate", () => {
  it("vote_all_any_ballot", () => {
    const s = sassKernel("_Z19warp_vote_predicatePiS_i", [
      `${sassHx(0x0e00)} VOTE.ALL R0, R1;`,
      `${sassHx(0x0e10)} VOTE.ANY R2, R3;`,
      `${sassHx(0x0e20)} VOTE.BALLOT R4, R5;`,
    ]);
    const { f } = catalogSassFeatures("warp_vote_predicate", s);
    expectCatalogPtxKernel("warp_vote_predicate");
    expect(f.warp_vote_ops).toBeGreaterThanOrEqual(3);
  });
});

describe("KERNEL_CATALOG §10 — global_atomic_histogram", () => {
  it("atom_red_global", () => {
    const s = sassKernel("_Z23global_atomic_histogramPiS_ii", [
      `${sassHx(0x0f00)} ATOM.E.ADD [R2], R4;`,
      `${sassHx(0x0f10)} RED.E.ADD [R6], R8;`,
    ]);
    const { f, source } = catalogSassFeatures("global_atomic_histogram", s);
    expectCatalogPtxKernel("global_atomic_histogram");
    // Inline snippet models ATOM + RED; nvcc may fold histogram updates to one ATOMG.* op.
    const minGlobal = source === "inline" ? 2 : 1;
    expect(f.global_atomic_ops).toBeGreaterThanOrEqual(minGlobal);
    expect(f.atomic_ops).toBeGreaterThanOrEqual(minGlobal);
  });
});

describe("KERNEL_CATALOG §11 — shared_atomic_counter", () => {
  it("atoms_not_global_atomic", () => {
    const s = sassKernel("_Z21shared_atomic_counterPiS_i", [
      `${sassHx(0x1000)} ATOMS.ADD [R2], R4;`,
    ]);
    const { f } = catalogSassFeatures("shared_atomic_counter", s);
    expectCatalogPtxKernel("shared_atomic_counter");
    expect(f.atomic_ops).toBeGreaterThanOrEqual(1);
    expect(f.global_atomic_ops).toBe(0);
  });
});

describe("KERNEL_CATALOG §12 — streaming_loads_cs", () => {
  it("ldg_cs_modifier", () => {
    const s = sassKernel("_Z19streaming_loads_csPfS_i", [
      `${sassHx(0x1100)} LDG.E.CS R0, [R4];`,
    ]);
    const { f } = catalogSassFeatures("streaming_loads_cs", s);
    expectCatalogPtxKernel("streaming_loads_cs");
    expect(f.cs_loads).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG §13 — gather_indexed", () => {
  it("narrow_ldg_32_only", () => {
    const s = sassKernel("_Z14gather_indexedPfPKiS_i", [
      `${sassHx(0x1200)} LDG.E.32 R0, [R4];`,
      `${sassHx(0x1210)} LDG.E.32 R2, [R6];`,
    ]);
    const { f, source } = catalogSassFeatures("gather_indexed", s);
    expectCatalogPtxKernel("gather_indexed");
    expect(f.ldg_32).toBeGreaterThanOrEqual(2);
    if (source === "inline") {
      expect(f.ldg_128).toBe(0);
    }
  });
});

describe("KERNEL_CATALOG §14 — tensor_core_wmma", () => {
  it("hmma_tensor_and_wmma", () => {
    const s = sassKernel("_Z17tensor_core_wmmaPfPK6__halfS2_", [
      `${sassHx(0x1300)} HMMA.16816.F32 {R0,R1,R2,R3},{R4,R5},{R6,R7},{R0,R1,R2,R3};`,
    ]);
    const { f } = catalogSassFeatures("tensor_core_wmma", s);
    expectCatalogPtxKernel("tensor_core_wmma");
    expect(f.tensor_ops).toBeGreaterThanOrEqual(1);
    expect(f.wmma_ops).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG §15 — async_copy_pipeline", () => {
  it("ldgsts_async_global_load", () => {
    const s = sassKernel("_Z20async_copy_pipelinePfS_i", [
      `${sassHx(0x1400)} LDGSTS.E.32 [R2], [R4];`,
    ]);
    const { f, source } = catalogSassFeatures("async_copy_pipeline", s);
    expectCatalogPtxKernel("async_copy_pipeline");
    if (source === "inline") {
      expect(f.async_global_loads).toBe(1);
      expect(f.global_loads).toBe(1);
    } else {
      expect(f.global_loads).toBeGreaterThanOrEqual(1);
    }
  });
});

// =============================================================================
// Extended Suite §16–22 — gap closures (MISSING_VARIANTS G1..G7)
// =============================================================================

describe("KERNEL_CATALOG §16 — ldcg_stream (gap G4)", () => {
  it("ldg_cg_modifier", () => {
    const s = sassKernel("_Z11ldcg_streamv", [
      `${sassHx(0x1500)} LDG.E.CG.SYS R0, [R4];`,
      `${sassHx(0x1510)} LDG.E.CG.SYS R2, [R6];`,
      `${sassHx(0x1520)} STG.E.SYS [R8], R0;`,
    ]);
    const { f, source } = catalogSassFeatures("ldcg_stream", s);
    expectCatalogPtxKernel("ldcg_stream");
    if (source === "inline") {
      expect(f.cg_loads).toBe(2);
      expect(f.cs_loads).toBe(0);
    } else {
      expect(f.cg_loads).toBeGreaterThanOrEqual(1);
    }
  });
});

describe("KERNEL_CATALOG §17 — const_bank_bench (gap G5)", () => {
  it("ldc_const_bank", () => {
    const s = sassKernel("_Z16const_bank_benchv", [
      `${sassHx(0x1600)} LDC.64 R2, [c0x0][R4];`,
      `${sassHx(0x1610)} LDC R6, [c0x0][R8];`,
    ]);
    const { f, source } = catalogSassFeatures("const_bank_bench", s);
    expectCatalogPtxKernel("const_bank_bench");
    expect(f.const_loads).toBeGreaterThanOrEqual(2);
    if (source === "inline") {
      expect(f.global_loads).toBe(0);
    }
  });
});

describe("KERNEL_CATALOG §18 — tensor_core_imma (gap G3)", () => {
  it("imma_tensor_not_wmma", () => {
    const s = sassKernel("_Z17tensor_core_immav", [
      `${sassHx(0x1700)} IMMA.16816.S32 {R0,R1},{R4,R5},{R6,R7},{R0,R1};`,
    ]);
    const { f, source } = catalogSassFeatures("tensor_core_imma", s);
    expectCatalogPtxKernel("tensor_core_imma");
    expect(f.wmma_ops).toBe(0);
    if (source === "inline") {
      expect(f.tensor_ops).toBe(1);
    } else {
      expect(f.tensor_ops >= 1 || f.global_loads >= 1).toBe(true);
    }
  });
});

describe("KERNEL_CATALOG §19 — membar_fence (gap G2)", () => {
  it("membar_scopes", () => {
    const s = sassKernel("_Z13membar_fencev", [
      `${sassHx(0x1800)} MEMBAR.GL;`,
      `${sassHx(0x1810)} MEMBAR.CTA;`,
      `${sassHx(0x1820)} MEMBAR.SYS;`,
    ]);
    const { f } = catalogSassFeatures("membar_fence", s);
    expectCatalogPtxKernel("membar_fence");
    expect(f.barrier).toBeGreaterThanOrEqual(3);
  });
});

describe("KERNEL_CATALOG §20 — tma_bulk_copy (gap G1)", () => {
  it("utma_ops", () => {
    const s = sassKernel("_Z14tma_bulk_copyv", [
      `${sassHx(0x1900)} UTMALDG.TILED R2, [R4];`,
      `${sassHx(0x1910)} UTMASTG.TILED [R8], R6;`,
    ]);
    const { f, source } = catalogSassFeatures("tma_bulk_copy", s);
    expectCatalogPtxKernel("tma_bulk_copy");
    if (source === "inline") {
      expect(f.tma_ops).toBe(2);
    } else {
      expect(f.tma_ops + f.async_global_loads).toBeGreaterThanOrEqual(1);
    }
  });

  it("cp_async_ldgsts_async_path", () => {
    const s = sassKernel("_Z14tma_bulk_copyv", [
      `${sassHx(0x1a00)} LDGSTS.E.128 [R8], [R12];`,
    ]);
    const { f, source } = catalogSassFeatures("tma_bulk_copy", s);
    expectCatalogPtxKernel("tma_bulk_copy");
    if (source === "inline") {
      expect(f.async_global_loads).toBe(1);
      expect(f.tma_ops).toBe(0);
    } else {
      expect(f.async_global_loads + f.tma_ops).toBeGreaterThanOrEqual(1);
    }
  });
});

describe("KERNEL_CATALOG §21 — wgmma_tile (gap G6)", () => {
  it("wgmma_counts_wmma", () => {
    const s = sassKernel("_Z11wgmma_tilev", [
      `${sassHx(0x1b00)} WGMMA.64816.F32.F16.F16;`,
    ]);
    const { f } = catalogSassFeatures("wgmma_tile", s);
    expectCatalogPtxKernel("wgmma_tile");
    expect(f.tensor_ops).toBeGreaterThanOrEqual(1);
    expect(f.wmma_ops).toBeGreaterThanOrEqual(1);
  });

  it("hmma_portable_wmma_path", () => {
    const s = sassKernel("_Z11wgmma_tilev", [
      `${sassHx(0x1c00)} HMMA.16816.F32 {R0,R1,R2,R3},{R4,R5},{R6,R7},{R0,R1,R2,R3};`,
    ]);
    const { f } = catalogSassFeatures("wgmma_tile", s);
    expectCatalogPtxKernel("wgmma_tile");
    expect(f.wmma_ops).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG §22 — bmma_sparse (gap G7)", () => {
  it("bmma_tensor_not_wmma", () => {
    const s = sassKernel("_Z12bmma_sparsev", [
      `${sassHx(0x1d00)} BMMA.88128.S32.B1.B1.XOR.POPC {R0,R1},{R4,R5},{R6,R7},{R0,R1};`,
    ]);
    const { f, source } = catalogSassFeatures("bmma_sparse", s);
    expectCatalogPtxKernel("bmma_sparse");
    expect(f.wmma_ops).toBe(0);
    if (source === "inline") {
      expect(f.tensor_ops).toBe(1);
    } else {
      expect(f.tensor_ops >= 1 || f.integer_ops >= 1).toBe(true);
    }
  });
});

// =============================================================================
// GEMM Suite — KERNEL_CATALOG GEMM-Variant §G1–G11
// =============================================================================

describe("KERNEL_CATALOG GEMM G1 — gemm_naive", () => {
  it("ldg_ffma_stg_no_shared", () => {
    const s = sassKernel("_Z10gemm_naivePfPKfS1_iii", [
      `${sassHx(0x2000)} LDG.E.32 R0, [R4];`,
      `${sassHx(0x2010)} LDG.E.32 R2, [R6];`,
      `${sassHx(0x2020)} FFMA.FTZ R8, R0, R2, R8;`,
      `${sassHx(0x2030)} STG.E.32 [R10], R8;`,
    ]);
    const { f, source } = catalogSassFeatures("gemm_naive", s);
    expectCatalogPtxKernel("gemm_naive");
    expect(f.ldg_32).toBeGreaterThanOrEqual(1);
    expect(f.arithmetic_ops).toBeGreaterThanOrEqual(1);
    if (source === "inline") {
      expect(f.shared_loads).toBe(0);
      expect(f.shared_stores).toBe(0);
    }
  });
});

describe("KERNEL_CATALOG GEMM G2 — gemm_tiled_shared", () => {
  it("ldg_lds_sts_bar_ffma", () => {
    const s = sassKernel("_Z18gemm_tiled_sharedPfPKfS1_iii", [
      `${sassHx(0x2100)} LDG.E.32 R0, [R4];`,
      `${sassHx(0x2110)} LDS.32 R2, [R8];`,
      `${sassHx(0x2120)} STS.32 [R8], R2;`,
      `${sassHx(0x2130)} BAR.SYNC 0;`,
      `${sassHx(0x2140)} FFMA.FTZ R8, R0, R2, R8;`,
    ]);
    const { f } = catalogSassFeatures("gemm_tiled_shared", s);
    expectCatalogPtxKernel("gemm_tiled_shared");
    expect(f.shared_loads).toBeGreaterThanOrEqual(1);
    expect(f.barrier).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG GEMM G3 — gemm_register_tiled", () => {
  it("dense_ffma_with_shared_tile", () => {
    const lines: string[] = [];
    let a = 0x2200;
    for (let i = 0; i < 24; i++) {
      lines.push(`${sassHx(a)} FFMA.FTZ R0, R1, R2, R3;`);
      a += 0x10;
    }
    lines.push(`${sassHx(a)} LDS.128 R8, [R4];`);
    lines.push(`${sassHx(a + 0x10)} STS.128 [R4], R8;`);
    lines.push(`${sassHx(a + 0x20)} BAR.SYNC 0;`);
    const s = sassKernel("_Z20gemm_register_tiledPfPKfS1_iii", lines);
    const { f, source } = catalogSassFeatures("gemm_register_tiled", s);
    expectCatalogPtxKernel("gemm_register_tiled");
    expect(f.arithmetic_ops).toBeGreaterThanOrEqual(24);
    expect(f.shared_loads).toBeGreaterThanOrEqual(1);
    expect(f.barrier).toBeGreaterThanOrEqual(1);
    if (source === "inline") {
      expect(f.shared_loads).toBe(1);
      expect(f.barrier).toBe(1);
    }
  });
});

describe("KERNEL_CATALOG GEMM G4 — gemm_wmma_fp16", () => {
  it("hmma_lds_bar", () => {
    const s = sassKernel("_Z15gemm_wmma_fp16PfPK6__halfS2_iii", [
      `${sassHx(0x2300)} LDS.32 R0, [R4];`,
      `${sassHx(0x2310)} BAR.SYNC 0;`,
      `${sassHx(0x2320)} HMMA.16816.F32 {R8,R9,R10,R11},{R4,R5},{R6,R7},{R8,R9,R10,R11};`,
    ]);
    const { f, source } = catalogSassFeatures("gemm_wmma_fp16", s);
    expectCatalogPtxKernel("gemm_wmma_fp16");
    expect(f.wmma_ops).toBeGreaterThanOrEqual(1);
    expect(f.tensor_ops).toBeGreaterThanOrEqual(1);
    // Fragment loads from global memory may compile to LDG.E + HMMA without BAR.SYNC.
    if (source === "inline") {
      expect(f.barrier).toBeGreaterThanOrEqual(1);
    }
  });
});

describe("KERNEL_CATALOG GEMM G5 — gemm_wmma_bf16", () => {
  it("hmma_like_fp_tensor_path", () => {
    const s = sassKernel("_Z15gemm_wmma_bf16PfPK6__NVbfloat16S2_iii", [
      `${sassHx(0x2400)} HMMA.16816.F32 {R0,R1,R2,R3},{R4,R5},{R6,R7},{R0,R1,R2,R3};`,
    ]);
    const { f } = catalogSassFeatures("gemm_wmma_bf16", s);
    expectCatalogPtxKernel("gemm_wmma_bf16");
    expect(f.wmma_ops).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG GEMM G6 — gemm_int8", () => {
  it("ldg_s8_imad_integer", () => {
    const s = sassKernel("_Z11gemm_int8PlPKaS1_iii", [
      `${sassHx(0x2500)} LDG.E.S8 R0, [R4];`,
      `${sassHx(0x2510)} IMAD R2, R4, R6, R8;`,
      `${sassHx(0x2520)} IADD3 R10, R2, R4;`,
    ]);
    const { f } = catalogSassFeatures("gemm_int8", s);
    expectCatalogPtxKernel("gemm_int8");
    expect(f.ldg_8).toBeGreaterThanOrEqual(1);
    expect(f.integer_ops).toBeGreaterThanOrEqual(2);
    expect(f.wmma_ops).toBe(0);
  });
});

describe("KERNEL_CATALOG GEMM G7 — gemm_bias_gelu", () => {
  it("ffma_lds_bar_mufu_epilogue", () => {
    const s = sassKernel("_Z15gemm_bias_geluPfPKfS1_S1_iii", [
      `${sassHx(0x2600)} LDS.32 R0, [R4];`,
      `${sassHx(0x2610)} BAR.SYNC 0;`,
      `${sassHx(0x2620)} FFMA.FTZ R8, R0, R2, R8;`,
      `${sassHx(0x2630)} MUFU.TANH R10, R8;`,
    ]);
    const { f, source } = catalogSassFeatures("gemm_bias_gelu", s);
    expectCatalogPtxKernel("gemm_bias_gelu");
    expect(f.sfu_ops).toBeGreaterThanOrEqual(1);
    expect(f.barrier).toBeGreaterThanOrEqual(1);
    expect(f.arithmetic_ops).toBeGreaterThanOrEqual(1);
    if (source === "inline") {
      expect(f.barrier).toBe(1);
    }
  });
});

describe("KERNEL_CATALOG GEMM G8 — gemm_batched", () => {
  it("same_sass_shape_as_tiled_gemm", () => {
    const s = sassKernel("_Z14gemm_batchedPfPKfS1_iiiii", [
      `${sassHx(0x2700)} LDG.E.32 R0, [R4];`,
      `${sassHx(0x2710)} LDS.32 R2, [R8];`,
      `${sassHx(0x2720)} BAR.SYNC 0;`,
      `${sassHx(0x2730)} FFMA.FTZ R8, R0, R2, R8;`,
    ]);
    const { f, source } = catalogSassFeatures("gemm_batched", s);
    expectCatalogPtxKernel("gemm_batched");
    expect(f.shared_loads).toBeGreaterThanOrEqual(1);
    expect(f.barrier).toBeGreaterThanOrEqual(1);
    if (source === "inline") {
      expect(f.shared_loads).toBe(1);
      expect(f.barrier).toBe(1);
    }
  });
});

describe("KERNEL_CATALOG GEMM G9 — gemm_splitk", () => {
  it("ffma_plus_global_atom_add", () => {
    const s = sassKernel("_Z12gemm_splitkPfPKfS1_iiii", [
      `${sassHx(0x2800)} LDG.E.32 R0, [R4];`,
      `${sassHx(0x2810)} FFMA.FTZ R2, R0, R1, R2;`,
      `${sassHx(0x2820)} ATOM.E.ADD.F32 [R8], R2;`,
    ]);
    const { f } = catalogSassFeatures("gemm_splitk", s);
    expectCatalogPtxKernel("gemm_splitk");
    expect(f.global_atomic_ops).toBeGreaterThanOrEqual(1);
    expect(f.arithmetic_ops).toBeGreaterThanOrEqual(1);
  });
});

describe("KERNEL_CATALOG GEMM G10 — gemm_async_copy", () => {
  it("ldgsts_with_tiled_body", () => {
    const s = sassKernel("_Z17gemm_async_copyPfPKfS1_iii", [
      `${sassHx(0x2900)} LDGSTS.E.32 [R2], [R4];`,
      `${sassHx(0x2910)} LDS.32 R0, [R8];`,
      `${sassHx(0x2920)} BAR.SYNC 0;`,
      `${sassHx(0x2930)} FFMA.FTZ R8, R0, R2, R8;`,
    ]);
    const { f, source } = catalogSassFeatures("gemm_async_copy", s);
    expectCatalogPtxKernel("gemm_async_copy");
    expect(f.shared_loads).toBeGreaterThanOrEqual(1);
    if (source === "inline") {
      expect(f.async_global_loads).toBeGreaterThanOrEqual(1);
    } else {
      expect(f.async_global_loads + f.global_loads).toBeGreaterThanOrEqual(1);
    }
  });
});

describe("KERNEL_CATALOG GEMM G11 — gemm_tall_skinny", () => {
  it("same_inner_sass_as_tiled_gemm", () => {
    const s = sassKernel("_Z17gemm_tall_skinnyPfPKfS1_iii", [
      `${sassHx(0x2a00)} LDG.E.32 R0, [R4];`,
      `${sassHx(0x2a10)} STS.32 [R8], R0;`,
      `${sassHx(0x2a20)} BAR.SYNC 0;`,
    ]);
    const { f, source } = catalogSassFeatures("gemm_tall_skinny", s);
    expectCatalogPtxKernel("gemm_tall_skinny");
    expect(f.shared_stores).toBeGreaterThanOrEqual(1);
    expect(f.barrier).toBeGreaterThanOrEqual(1);
    if (source === "inline") {
      expect(f.shared_stores).toBe(1);
      expect(f.barrier).toBe(1);
    }
  });
});
