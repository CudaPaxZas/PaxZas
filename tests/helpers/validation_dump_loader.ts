/**
 * Loads `cuobjdump` SASS / PTX artifacts for KERNEL_CATALOG tests.
 *
 * Resolution order (SASS):
 *   1. `PAXZAS_VALIDATION_SASS` — path to a `.sass` file or a directory of `.sass`
 *   2. `PAXZAS_VALIDATION_SASS_DIR` — directory of `*.sass`
 *   3. Default: `<PaxZas>/tests/data/sass/*.sass`
 *
 * PTX mirrors with `PAXZAS_VALIDATION_PTX`, `PAXZAS_VALIDATION_PTX_DIR`, and
 * `<PaxZas>/tests/data/ptx/*.ptx`.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { expect } from "vitest";
import { findKernelBody } from "../../src/analyzer/ptx_features";
import { extractSassFeatures } from "../../src/analyzer/sass_features";
import type { SassInstructionFeatures } from "../../src/analyzer/sass_features";

const testsDir = path.resolve(__dirname, "..");

function defaultSassDir(): string {
  return path.join(testsDir, "data", "sass");
}

function defaultPtxDir(): string {
  return path.join(testsDir, "data", "ptx");
}

function readTextFilesInDir(dir: string, ext: string): string[] {
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
    return [];
  }
  const names = fs
    .readdirSync(dir)
    .filter((n) => n.endsWith(ext))
    .sort();
  const out: string[] = [];
  for (const n of names) {
    const fp = path.join(dir, n);
    try {
      out.push(fs.readFileSync(fp, "utf8"));
    } catch {
      /* skip unreadable */
    }
  }
  return out;
}

function resolveSassModules(): string[] {
  const envPath = process.env.PAXZAS_VALIDATION_SASS;
  if (envPath) {
    if (!fs.existsSync(envPath)) {
      return [];
    }
    const st = fs.statSync(envPath);
    if (st.isFile()) {
      return [fs.readFileSync(envPath, "utf8")];
    }
    return readTextFilesInDir(envPath, ".sass");
  }
  const dir = process.env.PAXZAS_VALIDATION_SASS_DIR;
  if (dir && fs.existsSync(dir)) {
    return readTextFilesInDir(dir, ".sass");
  }
  return readTextFilesInDir(defaultSassDir(), ".sass");
}

function resolvePtxModules(): string[] {
  const envPath = process.env.PAXZAS_VALIDATION_PTX;
  if (envPath) {
    if (!fs.existsSync(envPath)) {
      return [];
    }
    const st = fs.statSync(envPath);
    if (st.isFile()) {
      return [fs.readFileSync(envPath, "utf8")];
    }
    return readTextFilesInDir(envPath, ".ptx");
  }
  const dir = process.env.PAXZAS_VALIDATION_PTX_DIR;
  if (dir && fs.existsSync(dir)) {
    return readTextFilesInDir(dir, ".ptx");
  }
  return readTextFilesInDir(defaultPtxDir(), ".ptx");
}

let cachedSass: string[] | undefined;
let cachedPtx: string[] | undefined;

export function getValidationSassModules(): string[] {
  if (cachedSass === undefined) {
    cachedSass = resolveSassModules();
  }
  return cachedSass;
}

export function getValidationPtxModules(): string[] {
  if (cachedPtx === undefined) {
    cachedPtx = resolvePtxModules();
  }
  return cachedPtx;
}

/**
 * Prefer real `cuobjdump --dump-sass` text: first file whose `Function :` name
 * matches `kernelSubstring`, else the inline one-kernel snippet (pass full
 * `sassKernel(...)` string).
 */
export function catalogSassFeatures(
  kernelSubstring: string,
  inlineCuobjdumpText: string
): { f: SassInstructionFeatures; source: "dump" | "inline" } {
  for (const text of getValidationSassModules()) {
    const [name, f] = extractSassFeatures(text, kernelSubstring);
    if (name) {
      return { f, source: "dump" };
    }
  }
  const [, f] = extractSassFeatures(inlineCuobjdumpText, undefined);
  return { f, source: "inline" };
}

/**
 * When PTX artifacts exist, require at least one module to contain a `.entry`
 * matching `kernelSubstring`. No-op if no PTX files (e.g. CI without dumps).
 */
export function expectCatalogPtxKernel(kernelSubstring: string): void {
  const modules = getValidationPtxModules();
  if (modules.length === 0) {
    return;
  }
  let hit = false;
  for (const m of modules) {
    const [, body] = findKernelBody(m, kernelSubstring);
    if (body && body.replace(/\s/g, "").length > 0) {
      hit = true;
      break;
    }
  }
  expect(
    hit,
    `expected PTX under tests/data/ptx (or PAXZAS_VALIDATION_PTX*) to define a kernel matching "${kernelSubstring}"`
  ).toBe(true);
}
