import { describe, expect, it } from "vitest";
import {
  allPresetMetadata,
  ccKeyFromPtxTarget,
  computeCapToArchKey,
  DEFAULT_FALLBACK_CC,
  getPresetMetadata,
  gpuSpecFromArch,
  GPU_SM_CONFIGS,
  inferSmCountFromGpuName,
  parsePtxSmTargetVersion,
  presetForSmVersion,
} from "../src/analyzer/gpu_spec";

describe("PTX .target → compute capability", () => {
  it("reads sm_XX from PTX", () => {
    expect(parsePtxSmTargetVersion(".target sm_89")).toBe(89);
    expect(parsePtxSmTargetVersion(".TARGET SM_80")).toBe(80);
  });

  it("maps sm versions to GPU_SM_CONFIGS keys", () => {
    expect(ccKeyFromPtxTarget(".target sm_80")).toBe("8.0");
    expect(ccKeyFromPtxTarget(".target sm_86")).toBe("8.6");
    expect(ccKeyFromPtxTarget(".target sm_89")).toBe("8.9");
    expect(ccKeyFromPtxTarget(".target sm_90")).toBe("9.0");
    expect(ccKeyFromPtxTarget(".target sm_75")).toBe("7.5");
    expect(ccKeyFromPtxTarget(".target sm_100")).toBe("10.0");
  });
});

describe("SMI compute_cap → arch key", () => {
  it("normalizes strings to GPU_SM_CONFIGS keys", () => {
    expect(computeCapToArchKey("8.9")).toBe("8.9");
    expect(computeCapToArchKey("8.90")).toBe("8.9");
    expect(computeCapToArchKey("9.0")).toBe("9.0");
    expect(computeCapToArchKey("10.0")).toBe("10.0");
    expect(computeCapToArchKey("10.00")).toBe("10.0");
  });
});

describe("gpuSpecFromArch", () => {
  it("merges SM count", () => {
    const g = gpuSpecFromArch("8.9", 128);
    expect(g).toMatchObject(GPU_SM_CONFIGS["8.9"]);
    expect(g.smCount).toBe(128);
  });

  it("default fallback CC is consumer Ampere table", () => {
    expect(DEFAULT_FALLBACK_CC).toBe("8.6");
    expect(GPU_SM_CONFIGS[DEFAULT_FALLBACK_CC].name).toBe("ampere-sm86");
  });

  it("exposes Blackwell CC 10.0 row", () => {
    expect(GPU_SM_CONFIGS["10.0"].name).toBe("blackwell-sm100");
  });

  it("includes Hopper and Blackwell presets", () => {
    const keys = allPresetMetadata().map((p) => p.key);
    expect(keys).toContain("h100-sxm");
    expect(keys).toContain("h200");
    expect(keys).toContain("b200");
    expect(keys).toContain("blackwell-consumer-default");
  });

  it("maps common SM targets to preset metadata", () => {
    expect(presetForSmVersion(90)?.key).toBe("h100-sxm");
    expect(presetForSmVersion(100)?.key).toBe("b200");
    expect(presetForSmVersion(120)?.key).toBe("blackwell-consumer-default");
    expect(getPresetMetadata("h200").cc).toBe("9.0");
  });
});

describe("B3 – inferSmCountFromGpuName H100 SKU disambiguation", () => {
  it("H100 SXM (generic name) → 132 SMs", () => {
    // "NVIDIA H100 80GB HBM3" is the typical nvidia-smi name for SXM5 nodes
    expect(inferSmCountFromGpuName("NVIDIA H100 80GB HBM3")).toBe(132);
    expect(inferSmCountFromGpuName("NVIDIA H100 SXM5 80GB")).toBe(132);
    expect(inferSmCountFromGpuName("H100")).toBe(132);
  });

  it("H100 PCIe → 114 SMs", () => {
    expect(inferSmCountFromGpuName("NVIDIA H100 PCIe")).toBe(114);
    expect(inferSmCountFromGpuName("NVIDIA H100 80GB PCIe")).toBe(114);
  });

  it("H100 NVL → 114 SMs", () => {
    expect(inferSmCountFromGpuName("NVIDIA H100 NVL")).toBe(114);
    expect(inferSmCountFromGpuName("NVIDIA H100 94GB NVL")).toBe(114);
  });

  it("H200 → 132 SMs (same die as H100 SXM)", () => {
    expect(inferSmCountFromGpuName("NVIDIA H200")).toBe(132);
    expect(inferSmCountFromGpuName("NVIDIA H200 SXM5 141GB")).toBe(132);
  });

  it("A100 / B200 are unaffected", () => {
    expect(inferSmCountFromGpuName("NVIDIA A100-SXM4-80GB")).toBe(108);
    expect(inferSmCountFromGpuName("NVIDIA B200")).toBe(192);
  });

  it("unknown device → undefined", () => {
    expect(inferSmCountFromGpuName("NVIDIA GeForce GTX 1080 Ti")).toBeUndefined();
  });
});
