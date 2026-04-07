/**
 * Mirrors tests/test_occupancy_model.py
 */
import { describe, expect, it } from "vitest";
import { AMPERE_LIKE_DEFAULT } from "../src/analyzer/gpu_spec";
import { analyzeOccupancyModel } from "../src/analyzer/occupancy_model";
import { mergeLaunchWithHints } from "../src/analyzer/merge_launch";
import { extractSassFeatures } from "../src/analyzer/sass_features";
import { PTX_HEAD } from "./helpers/ir_fixtures";

function inferRegsFromSass(sassText: string, kernel: string): number {
  const [, sass] = extractSassFeatures(sassText, kernel);
  if (sass.max_register_index < 0) {
    throw new Error("SASS fixture must mention registers");
  }
  return sass.max_register_index + 1;
}

describe("occupancy_model (Python parity)", () => {
  it("class_and_sources", () => {
    const ptx =
      PTX_HEAD +
      `
.visible .entry _Z8occ_high(
  .maxntid 128, 1, 1
)
{
  .shared .align 16 .b8 pool[4096];
  ret;
}
`;
    const sass = ["Function : _Z8occ_high", "    /*0100*/ LDG.E.32 R63, [R2];"].join(
      "\n"
    );
    const kernel = "_Z8occ_high";
    const sassRegs = inferRegsFromSass(sass, kernel);
    const merged = mergeLaunchWithHints(ptx, kernel, { threads: 128 }, sassRegs);

    expect(merged.threads).toBe(128);
    expect(merged.shared).toBe(4096);
    expect(merged.registers).toBe(64);

    const out = analyzeOccupancyModel(
      merged.threads,
      merged.shared,
      merged.registers,
      AMPERE_LIKE_DEFAULT,
      merged.threadsSource,
      merged.sharedSource,
      merged.registerSource
    );
    expect(["low", "medium", "high"]).toContain(out.class);
    expect(out.sources.threads).toBe("launch");
    expect(out.sources.shared).toBe("ptx.static_shared");
    expect(out.sources.registers).toBe("sass.inferred");
    expect(out.confidence).toBeGreaterThanOrEqual(0.0);
    expect(out.confidence).toBeLessThanOrEqual(1.0);
  });

  it("low_class_case", () => {
    const ptx =
      PTX_HEAD +
      `
.visible .entry _Z7occ_low(
  .maxntid 64, 1, 1
  .maxnreg 256
)
{
  ret;
}
`;
    const kernel = "_Z7occ_low";
    const merged = mergeLaunchWithHints(ptx, kernel, {}, undefined);

    expect(merged.threads).toBe(64);
    expect(merged.shared).toBe(0);
    expect(merged.registers).toBe(256);
    expect(merged.threadsSource).toBe("ptx.maxntid");
    expect(merged.sharedSource).toBe("ptx.static_shared");
    expect(merged.registerSource).toBe("ptx.maxnreg");

    const out = analyzeOccupancyModel(
      merged.threads,
      merged.shared,
      merged.registers,
      AMPERE_LIKE_DEFAULT,
      merged.threadsSource,
      merged.sharedSource,
      merged.registerSource
    );
    expect(["low", "medium"]).toContain(out.class);
  });
});
