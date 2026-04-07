/**
 * Mirrors tests/test_sass_features.py when fixture file is absent:
 * uses minimal inline SASS with expected opcode classes.
 */
import { describe, expect, it } from "vitest";
import {
  extractSassFeatures,
  inferRegistersPerThreadFromSass,
} from "../src/analyzer/sass_features";
import { sassHx } from "./helpers/ir_fixtures";

/** Minimal dump shaped like cuobjdump (matches _SASS_OPCODE_RE). */
function inlineSassFixture(kernel: string): string {
  const lines: string[] = [`Function : ${kernel}`, ""];
  let a = 0x0100;
  lines.push(`${sassHx(a)} STG.E.SYS [R2], R4;`);
  a += 0x10;
  lines.push(`${sassHx(a)} IADD R0, R1, R2;`);
  a += 0x10;
  lines.push(`${sassHx(a)} BRA 0x200;`);
  return lines.join("\n");
}

const KERNEL = "_ZN4test7kernels6sampleEv";

describe("sass_features (Python parity shape)", () => {
  it("extract_sass_features_counts_basic_op_classes", () => {
    const text = inlineSassFixture(KERNEL);
    const [name, f] = extractSassFeatures(text, KERNEL);
    expect(name).toBe(KERNEL);
    expect(f.global_loads).toBe(0);
    expect(f.global_stores).toBe(1);
    expect(f.shared_loads).toBe(0);
    expect(f.shared_stores).toBe(0);
    expect(f.arithmetic_ops).toBeGreaterThanOrEqual(1);
    expect(f.tensor_ops).toBe(0);
    expect(f.barrier).toBe(0);
    expect(f.branch).toBeGreaterThanOrEqual(1);
  });

  it("infer_registers_per_thread_from_sass", () => {
    const text = inlineSassFixture(KERNEL);
    const regs = inferRegistersPerThreadFromSass(text, KERNEL);
    expect(regs).toBeDefined();
    expect(regs!).toBeGreaterThanOrEqual(1);
  });
});
