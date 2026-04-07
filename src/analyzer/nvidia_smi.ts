/**
 * Best-effort probe of the default NVIDIA GPU via nvidia-smi (no native bindings).
 *
 * GUI-launch often omits CUDA dirs from PATH; we try known install locations and
 * augment PATH for the child.
 */

import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/** Prefer MP count; fall back for older drivers that reject the field. */
const NVSMI_QUERY_TRIES = [
  [
    "--query-gpu=name,compute_cap,multiprocessor_count",
    "--format=csv,noheader",
  ],
  ["--query-gpu=name,compute_cap", "--format=csv,noheader"],
] as const;

export interface SmiGpuRow {
  name: string;
  compute_cap: string;
  multiprocessor_count?: number;
}

export const RESOLUTION_SMI_NOT_PRESENT =
  "auto:ampere-sm86 — nvidia-smi not present.";

export const RESOLUTION_SMI_PROBE_FAILED =
  "auto:ampere-sm86 — nvidia-smi did not return usable GPU data.";

export type NvidiaSmiProbeResult =
  | { ok: true; row: SmiGpuRow }
  | { ok: false; reason: "not_present" | "failed" };

function errnoCode(e: unknown): string | undefined {
  if (typeof e === "object" && e !== null && "code" in e) {
    const c = (e as { code?: string }).code;
    return typeof c === "string" ? c : undefined;
  }
  return undefined;
}

function stripQuotes(s: string): string {
  const t = s.trim();
  if (t.startsWith('"') && t.endsWith('"')) {
    return t.slice(1, -1);
  }
  return t;
}

const CC_RE = /^\d+\.\d+$/;

/** Parse 2- or 3-field nvidia-smi CSV (commas inside quoted GPU name). */
export function parseSmiGpuCsvLine(line: string): SmiGpuRow | undefined {
  const trimmed = line.trim();
  if (!trimmed) {
    return undefined;
  }
  const last = trimmed.lastIndexOf(",");
  if (last < 0) {
    return undefined;
  }
  const tail = trimmed.slice(last + 1).trim();
  const beforeRest = trimmed.slice(0, last);
  const mid = beforeRest.lastIndexOf(",");
  if (mid < 0) {
    if (!CC_RE.test(tail)) {
      return undefined;
    }
    return {
      name: stripQuotes(beforeRest),
      compute_cap: tail,
    };
  }
  const maybeCc = beforeRest.slice(mid + 1).trim();
  const namePart = beforeRest.slice(0, mid).trim();
  if (!CC_RE.test(maybeCc)) {
    return undefined;
  }
  const mp = parseInt(tail, 10);
  const multiprocessor_count = Number.isFinite(mp) ? mp : undefined;
  return {
    name: stripQuotes(namePart),
    compute_cap: maybeCc,
    multiprocessor_count,
  };
}

/** Ordered list of executables to try (PATH + common install paths). */
export function nvidiaSmiCandidates(): string[] {
  const out: string[] = [];
  const cudaRoot = process.env.CUDA_PATH ?? process.env.CUDA_HOME;
  if (cudaRoot?.trim()) {
    const r = cudaRoot.trim();
    if (process.platform === "win32") {
      out.push(`${r}\\bin\\nvidia-smi.exe`);
    } else {
      out.push(`${r}/bin/nvidia-smi`);
    }
  }
  if (process.platform === "win32") {
    out.push(
      "nvidia-smi",
      String.raw`C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe`
    );
  } else {
    out.push(
      "/usr/bin/nvidia-smi",
      "/usr/local/cuda/bin/nvidia-smi",
      "/opt/cuda/bin/nvidia-smi",
      "nvidia-smi"
    );
  }
  return out;
}

export function envForNvidiaSmi(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  const sep = process.platform === "win32" ? ";" : ":";
  const prepend: string[] = [];
  const cudaRoot = process.env.CUDA_PATH ?? process.env.CUDA_HOME;
  if (cudaRoot?.trim()) {
    prepend.push(
      process.platform === "win32"
        ? `${cudaRoot.trim()}\\bin`
        : `${cudaRoot.trim()}/bin`
    );
  }
  if (process.platform === "win32") {
    prepend.push(String.raw`C:\Program Files\NVIDIA Corporation\NVSMI`);
  } else {
    prepend.push("/usr/local/cuda/bin", "/opt/cuda/bin", "/usr/bin");
  }
  const cur =
    (process.platform === "win32"
      ? env.Path ?? env.PATH
      : env.PATH) ?? "";
  const merged = [...prepend, cur].filter(Boolean).join(sep);
  env.PATH = merged;
  env.Path = merged;
  return env;
}

function parseFirstGpuStdout(stdout: string): SmiGpuRow | undefined {
  const line = stdout
    .split(/\r?\n/)
    .map((l) => l.trim())
    .find((l) => l.length > 0);
  return line ? parseSmiGpuCsvLine(line) : undefined;
}

export async function queryFirstGpuFromNvidiaSmi(): Promise<NvidiaSmiProbeResult> {
  const childEnv = envForNvidiaSmi();
  let anyRan = false;

  for (const bin of nvidiaSmiCandidates()) {
    for (const q of NVSMI_QUERY_TRIES) {
      try {
        const { stdout } = await execFileAsync(bin, [...q], {
          timeout: 8000,
          windowsHide: true,
          maxBuffer: 256 * 1024,
          env: childEnv,
        });
        anyRan = true;
        const row = parseFirstGpuStdout(stdout);
        if (row) {
          return { ok: true, row };
        }
      } catch (e) {
        if (errnoCode(e) !== "ENOENT") {
          anyRan = true;
        }
      }
    }
  }

  if (!anyRan) {
    return { ok: false, reason: "not_present" };
  }
  return { ok: false, reason: "failed" };
}
