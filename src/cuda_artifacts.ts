/**
 * Discover CUDA build artifacts in the workspace (2Instructions.md Step 1).
 * Uses ripgrep-backed findFiles instead of synchronous directory walks.
 */

import * as vscode from "vscode";

const EXCLUDE =
  "**/{node_modules,.git,.hg,.svn,out,dist,build/Release,build/Debug}/**";

const MAX_PER_PATTERN = 4000;

export async function findCudaArtifactUris(): Promise<vscode.Uri[]> {
  const patterns = [
    "**/*.ptx",
    "**/*.sass",
    "**/*.PTX",
    "**/*.SASS",
  ];
  const batches = await Promise.all(
    patterns.map((p) => vscode.workspace.findFiles(p, EXCLUDE, MAX_PER_PATTERN))
  );
  const seen = new Map<string, vscode.Uri>();
  for (const batch of batches) {
    for (const u of batch) {
      seen.set(u.fsPath, u);
    }
  }
  return [...seen.values()].sort((a, b) =>
    a.fsPath.localeCompare(b.fsPath, undefined, { sensitivity: "base" })
  );
}
