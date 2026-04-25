/**
 * PTX entry and kernel hint parsing (matches ptx_parse.py).
 */

export interface PtxKernelHints {
  kernelName: string | undefined;
  maxnreg: number | undefined;
  maxntid: [number, number, number] | undefined;
  staticSharedBytes: number;
  dynamicSharedDetected: boolean;
}

const ENTRY_RE = /\.(?:visible\s+)?entry\s+(\S+)\s*\(/g;
// B8: anchored to line-start (multiline `m` flag) with optional leading whitespace.
// Without anchoring, these patterns would match `.maxnreg` inside `//` comments
// (e.g. "// Set .maxnreg 48") or inside hypothetical extended directives like
// `.maxnreg_v2 64`.  Requiring `^[ \t]*` ensures only true standalone directives
// on their own lines are captured.
const MAXNREG_RE = /^[ \t]*\.maxnreg\s+(\d+)/m;
const MAXNTID_RE = /^[ \t]*\.maxntid\s+(\d+)(?:\s*,\s*(\d+)(?:\s*,\s*(\d+))?)?/m;
// Group 1: bit-width (8/16/32/64) from the type qualifier — may be absent.
// Group 2: element count in brackets.
// When no type qualifier is present the declaration is treated as raw bytes
// (width = 8 bits = 1 byte), which matches how nvcc emits un-typed .shared regions.
const SHARED_ARRAY_RE =
  /\.shared(?:\s+\.align\s+\d+)?(?:\s+\.(?:b|s|u|f)(8|16|32|64))?\s+\S+\s*\[\s*(\d+)\s*\]/g;
const DYNAMIC_SHARED_RE =
  /(?:\.extern\s+\.shared\b|\.shared(?:\s+\.align\s+\d+)?(?:\s+\.(?:b|s|u|f)(?:8|16|32|64))?\s+\S+\s*\[\s*\])/;

interface EntryMatch {
  name: string;
  start: number;
  parenEnd: number;
}

function findAllEntries(ptx: string): EntryMatch[] {
  const out: EntryMatch[] = [];
  ENTRY_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = ENTRY_RE.exec(ptx)) !== null) {
    out.push({ name: m[1]!, start: m.index, parenEnd: m.index + m[0].length });
  }
  return out;
}

/** Returns (kernel_name, match_start, paren_end) paren_end after `(` */
export function pickEntry(
  ptx: string,
  kernelSubstring: string | undefined
): { name: string; start: number; parenEnd: number } | undefined {
  const matches = findAllEntries(ptx);
  if (matches.length === 0) {
    return undefined;
  }
  if (kernelSubstring === undefined) {
    const m = matches[0]!;
    return { name: m.name, start: m.start, parenEnd: m.parenEnd };
  }
  const exact = matches.filter((e) => e.name === kernelSubstring);
  if (exact.length > 0) {
    const m = exact[0]!;
    return { name: m.name, start: m.start, parenEnd: m.parenEnd };
  }
  const partial = matches.filter((e) => e.name.includes(kernelSubstring));
  if (partial.length > 0) {
    const m = partial[0]!;
    return { name: m.name, start: m.start, parenEnd: m.parenEnd };
  }
  return undefined;
}

function entryChunk(
  ptx: string,
  entryMatchEnd: number
): { directiveRegion: string; sharedScan: string } {
  const sliceEnd = Math.min(ptx.length, entryMatchEnd + 65536);
  const chunk = ptx.slice(entryMatchEnd, sliceEnd);
  const brace = chunk.indexOf("{");
  if (brace === -1) {
    return { directiveRegion: chunk, sharedScan: chunk };
  }
  const directive = chunk.slice(0, brace);
  const bodyPrefix = Math.min(chunk.length, brace + 1 + 8192);
  const sharedScan = chunk.slice(0, bodyPrefix);
  return { directiveRegion: directive, sharedScan };
}

function estimateStaticSharedBytes(ptxSnippet: string): number {
  let total = 0;
  SHARED_ARRAY_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = SHARED_ARRAY_RE.exec(ptxSnippet)) !== null) {
    // m[1] = bit-width string ("8", "16", "32", "64") or undefined when absent.
    // Absent type → treat as raw bytes (1 byte per element), matching how nvcc
    // emits un-typed or .b8 shared regions.
    const bits = m[1] !== undefined ? parseInt(m[1], 10) : 8;
    const byteWidth = bits / 8;
    const count = parseInt(m[2]!, 10);
    total += count * byteWidth;
  }
  return total;
}

function detectDynamicShared(ptxSnippet: string): boolean {
  return DYNAMIC_SHARED_RE.test(ptxSnippet);
}

export function parsePtxKernelHints(
  ptx: string,
  kernelSubstring: string | undefined
): PtxKernelHints {
  const picked = pickEntry(ptx, kernelSubstring);
  if (!picked) {
    return {
      kernelName: undefined,
      maxnreg: undefined,
      maxntid: undefined,
      staticSharedBytes: 0,
      dynamicSharedDetected: false,
    };
  }
  const { directiveRegion, sharedScan } = entryChunk(ptx, picked.parenEnd);

  const maxnregM = MAXNREG_RE.exec(directiveRegion);
  const maxnreg = maxnregM ? parseInt(maxnregM[1]!, 10) : undefined;

  let maxntid: [number, number, number] | undefined;
  const maxntidM = MAXNTID_RE.exec(directiveRegion);
  if (maxntidM) {
    maxntid = [
      parseInt(maxntidM[1]!, 10),
      maxntidM[2] !== undefined ? parseInt(maxntidM[2], 10) : 1,
      maxntidM[3] !== undefined ? parseInt(maxntidM[3], 10) : 1,
    ];
  }

  const shared = estimateStaticSharedBytes(sharedScan);
  const dynamicShared = detectDynamicShared(sharedScan);
  return {
    kernelName: picked.name,
    maxnreg,
    maxntid,
    staticSharedBytes: shared,
    dynamicSharedDetected: dynamicShared,
  };
}

export function maxThreadsFromMaxntid(
  triple: [number, number, number] | undefined
): number | undefined {
  if (!triple) {
    return undefined;
  }
  const [x, y, z] = triple;
  return x * y * z;
}

export function listPtxEntryNames(ptx: string): string[] {
  return findAllEntries(ptx).map((e) => e.name);
}
