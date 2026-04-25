import { describe, expect, it } from "vitest";
import { inferSmCountFromGpuName } from "../src/analyzer/gpu_spec";

describe("gpu_spec SM-count name fallback", () => {
  it("infers known model names", () => {
    expect(inferSmCountFromGpuName("NVIDIA GeForce RTX 4090")).toBe(128);
    expect(inferSmCountFromGpuName("NVIDIA A100-SXM4-80GB")).toBe(108);
    expect(inferSmCountFromGpuName("NVIDIA H100 PCIe")).toBe(114); // 114 SMs — GH100 with 2 disabled GPCs
  });

  it("returns undefined for unknown model names", () => {
    expect(inferSmCountFromGpuName("Mystery Accelerator 1234")).toBeUndefined();
    expect(inferSmCountFromGpuName("  ")).toBeUndefined();
  });
});
