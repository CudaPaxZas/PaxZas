import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const GPU_INFO_CACHE = path.join(process.env.USERPROFILE ?? process.env.HOME ?? ".", ".paxzas_gpuinfo.json");

interface GpuInfoCache {
  name: string;
  computeCap: string;
  smCount: number;
  maxThreadsPerBlock: number;
  updatedAt: string;
}

export interface LocalCudaDetectResult {
  smCount: number;
  source: "local.cache" | "local.cuda.detect";
}

export interface LocalCudaDetectOptions {
  expectedName?: string;
  expectedComputeCap?: string;
}

const DETECT_CU = `
#include <cstdio>
#include <cuda_runtime.h>

int main() {
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess || count <= 0) {
    return 2;
  }

  cudaDeviceProp p{};
  e = cudaGetDeviceProperties(&p, 0);
  if (e != cudaSuccess) {
    return 3;
  }

  std::printf("NAME=%s\\n", p.name);
  std::printf("COMPUTE_CAP=%d.%d\\n", p.major, p.minor);
  std::printf("SM_COUNT=%d\\n", p.multiProcessorCount);
  std::printf("MAX_THREADS_PER_BLOCK=%d\\n", p.maxThreadsPerBlock);
  return 0;
}
`;

function parseDetectOutput(stdout: string): GpuInfoCache | undefined {
  const lines = stdout
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  let name: string | undefined;
  let computeCap: string | undefined;
  let smCount: number | undefined;
  let maxThreadsPerBlock: number | undefined;

  for (const line of lines) {
    const idx = line.indexOf("=");
    if (idx <= 0) {
      continue;
    }
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim();
    if (key === "NAME") {
      name = value;
    } else if (key === "COMPUTE_CAP") {
      computeCap = value;
    } else if (key === "SM_COUNT") {
      const n = parseInt(value, 10);
      if (Number.isFinite(n) && n > 0) {
        smCount = n;
      }
    } else if (key === "MAX_THREADS_PER_BLOCK") {
      const n = parseInt(value, 10);
      if (Number.isFinite(n) && n > 0) {
        maxThreadsPerBlock = n;
      }
    }
  }

  if (!name || !computeCap || smCount === undefined || maxThreadsPerBlock === undefined) {
    return undefined;
  }

  return {
    name,
    computeCap,
    smCount,
    maxThreadsPerBlock,
    updatedAt: new Date().toISOString(),
  };
}

async function readCache(): Promise<GpuInfoCache | undefined> {
  try {
    const raw = await fs.readFile(GPU_INFO_CACHE, "utf-8");
    const parsed = JSON.parse(raw) as Partial<GpuInfoCache>;
    if (
      typeof parsed.name === "string" &&
      typeof parsed.computeCap === "string" &&
      typeof parsed.smCount === "number" &&
      Number.isFinite(parsed.smCount) &&
      parsed.smCount > 0 &&
      typeof parsed.maxThreadsPerBlock === "number" &&
      Number.isFinite(parsed.maxThreadsPerBlock) &&
      parsed.maxThreadsPerBlock > 0
    ) {
      return {
        name: parsed.name,
        computeCap: parsed.computeCap,
        smCount: parsed.smCount,
        maxThreadsPerBlock: parsed.maxThreadsPerBlock,
        updatedAt:
          typeof parsed.updatedAt === "string"
            ? parsed.updatedAt
            : new Date().toISOString(),
      };
    }
  } catch {
    return undefined;
  }
  return undefined;
}

async function writeCache(info: GpuInfoCache): Promise<void> {
  const payload = JSON.stringify(info, null, 2);
  await fs.writeFile(GPU_INFO_CACHE, payload, "utf-8");
}

function versionWeight(v: string): number[] {
  return v
    .split(".")
    .map((p) => parseInt(p, 10))
    .filter((n) => Number.isFinite(n));
}

function compareVersionDesc(a: string, b: string): number {
  const va = versionWeight(a);
  const vb = versionWeight(b);
  const n = Math.max(va.length, vb.length);
  for (let i = 0; i < n; i += 1) {
    const da = va[i] ?? 0;
    const db = vb[i] ?? 0;
    if (da !== db) {
      return db - da;
    }
  }
  return 0;
}

async function nvccCandidates(): Promise<string[]> {
  const out: string[] = [];
  const cudaRoot = process.env.CUDA_PATH ?? process.env.CUDA_HOME;
  if (cudaRoot?.trim()) {
    out.push(
      process.platform === "win32"
        ? path.join(cudaRoot.trim(), "bin", "nvcc.exe")
        : path.join(cudaRoot.trim(), "bin", "nvcc")
    );
  }

  if (process.platform === "win32") {
    const base = String.raw`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`;
    try {
      const dirs = await fs.readdir(base, { withFileTypes: true });
      const versions = dirs
        .filter((d) => d.isDirectory())
        .map((d) => d.name)
        .sort(compareVersionDesc);
      for (const v of versions) {
        out.push(path.join(base, v, "bin", "nvcc.exe"));
      }
    } catch {
      // ignore
    }
  } else {
    out.push("/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc");
  }

  out.push("nvcc");
  return Array.from(new Set(out));
}

function cacheMatches(cache: GpuInfoCache, options: LocalCudaDetectOptions): boolean {
  const expectedName = options.expectedName?.trim();
  const expectedCc = options.expectedComputeCap?.trim();
  if (expectedName && cache.name !== expectedName) {
    return false;
  }
  if (expectedCc && cache.computeCap !== expectedCc) {
    return false;
  }
  return true;
}

export async function detectSmCountLocally(
  options: LocalCudaDetectOptions = {}
): Promise<LocalCudaDetectResult | undefined> {
  const cached = await readCache();
  if (cached && cacheMatches(cached, options)) {
    return { smCount: cached.smCount, source: "local.cache" };
  }

  const workDir = await fs.mkdtemp(path.join(tmpdir(), "paxzas-gpu-detect-"));
  const cuPath = path.join(workDir, "detect_gpu.cu");
  const exePath = path.join(
    workDir,
    process.platform === "win32" ? "detect_gpu.exe" : "detect_gpu"
  );

  try {
    await fs.writeFile(cuPath, DETECT_CU, "utf-8");
    const candidates = await nvccCandidates();

    for (const nvcc of candidates) {
      try {
        await execFileAsync(nvcc, [cuPath, "-o", exePath], {
          timeout: 60_000,
          windowsHide: true,
          maxBuffer: 512 * 1024,
        });
        const { stdout } = await execFileAsync(exePath, [], {
          timeout: 10_000,
          windowsHide: true,
          maxBuffer: 128 * 1024,
        });
        const parsed = parseDetectOutput(stdout);
        if (!parsed) {
          continue;
        }
        await writeCache(parsed);
        return { smCount: parsed.smCount, source: "local.cuda.detect" };
      } catch {
        // Try next nvcc candidate.
      }
    }
  } finally {
    try {
      await fs.rm(workDir, { recursive: true, force: true });
    } catch {
      // ignore cleanup errors
    }
  }

  return undefined;
}
