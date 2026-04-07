import { describe, expect, it } from "vitest";
import { parseSmiGpuCsvLine } from "../src/analyzer/nvidia_smi";

describe("parseSmiGpuCsvLine", () => {
  it("parses 2 columns", () => {
    expect(parseSmiGpuCsvLine("NVIDIA RTX 4090, 8.9")).toEqual({
      name: "NVIDIA RTX 4090",
      compute_cap: "8.9",
    });
  });

  it("parses 3 columns with multiprocessor_count", () => {
    expect(
      parseSmiGpuCsvLine("NVIDIA RTX 4090, 8.9, 128")
    ).toEqual({
      name: "NVIDIA RTX 4090",
      compute_cap: "8.9",
      multiprocessor_count: 128,
    });
  });

  it("handles quoted GPU name with comma", () => {
    expect(
      parseSmiGpuCsvLine('"ACME, RTX", 8.6, 84')
    ).toEqual({
      name: "ACME, RTX",
      compute_cap: "8.6",
      multiprocessor_count: 84,
    });
  });
});
