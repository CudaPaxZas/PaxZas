/**
 * Forces every nvidia-smi spawn to ENOENT so auto resolution is deterministic
 * and you can see the same line as in the extension Output channel.
 */
import { describe, expect, it, vi } from "vitest";

vi.mock("node:child_process", () => ({
  execFile: vi.fn(
    (
      _cmd: string,
      _args: readonly string[],
      _opts: object,
      cb: (
        err: NodeJS.ErrnoException | null,
        stdout?: string,
        stderr?: string
      ) => void
    ) => {
      const err = new Error("spawn ENOENT") as NodeJS.ErrnoException;
      err.code = "ENOENT";
      queueMicrotask(() => cb(err, "", ""));
    }
  ),
}));

import { resolveGpuSpecForAnalysis } from "../src/analyzer/gpu_spec";
import { RESOLUTION_SMI_NOT_PRESENT } from "../src/analyzer/nvidia_smi";

describe("nvidia-smi not available (mocked ENOENT)", () => {
  it("prints the single static GPU spec line", async () => {
    const resolved = await resolveGpuSpecForAnalysis("auto");
    // Shown in terminal when you run: npx vitest run tests/nvidia_smi_probe.test.ts
    console.log("[CUDA Analyzer GPU spec]", resolved.resolution);
    expect(resolved.resolution).toBe(RESOLUTION_SMI_NOT_PRESENT);
    expect(resolved.spec.name).toBe("ampere-sm86");
  });
});
