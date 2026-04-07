import { describe, expect, it } from "vitest";
import {
  RESOLUTION_SMI_NOT_PRESENT,
  RESOLUTION_SMI_PROBE_FAILED,
} from "../src/analyzer/nvidia_smi";

describe("nvidia-smi fallback messages (fixed copy)", () => {
  it("uses stable wording for missing binary", () => {
    expect(RESOLUTION_SMI_NOT_PRESENT).toBe(
      "auto:ampere-sm86 — nvidia-smi not present."
    );
  });

  it("uses stable wording for failed probe", () => {
    expect(RESOLUTION_SMI_PROBE_FAILED).toBe(
      "auto:ampere-sm86 — nvidia-smi did not return usable GPU data."
    );
  });
});
