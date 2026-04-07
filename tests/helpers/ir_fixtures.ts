/**
 * Mirrors tests/ir_fixtures.py — exercise real TS extractors.
 */

import {
  extractInstructionFeatures,
  findKernelBody,
} from "../../src/analyzer/ptx_features";
import { extractSassFeatures } from "../../src/analyzer/sass_features";
import type { PtxInstructionFeatures } from "../../src/analyzer/ptx_features";
import type { SassInstructionFeatures } from "../../src/analyzer/sass_features";

export const PTX_HEAD = `\
.version 7.4
.target sm_75
.address_size 64
`;

export function sassHx(addr: number): string {
  return `    /*${addr.toString(16).padStart(4, "0").toLowerCase()}*/`;
}

export function featuresFromPtx(
  ptxModule: string,
  kernelSubstring: string | undefined = undefined
): PtxInstructionFeatures {
  const [, body] = findKernelBody(ptxModule, kernelSubstring);
  if (!body) {
    throw new Error("expected kernel body");
  }
  return extractInstructionFeatures(body);
}

export function featuresFromSass(
  sassModule: string,
  kernelSubstring: string | undefined = undefined
): SassInstructionFeatures {
  const [, sass] = extractSassFeatures(sassModule, kernelSubstring);
  return sass;
}
