/**
 * Discover CUDA build artifacts in the workspace (2Instructions.md Step 1).
 * Uses ripgrep-backed findFiles instead of synchronous directory walks.
 */

import * as vscode from "vscode";
import * as path from "path";

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

function stripHashSuffix(stem: string): string {
  return stem.replace(/_[0-9a-f]{8,}$/i, "");
}

function isSassLikeExt(ext: string): boolean {
  const e = ext.toLowerCase();
  return e === ".sass" || e === ".s";
}

/**
 * Locate a sidecar SASS file for a PTX file in the same directory.
 * Priority: exact stem match, then stem with trailing hash removed.
 */
export async function findSupplementalSassForPtx(
  ptxUri: vscode.Uri
): Promise<vscode.Uri | undefined> {
  const ptxExt = path.extname(ptxUri.fsPath).toLowerCase();
  if (ptxExt !== ".ptx") {
    return undefined;
  }

  const dir = path.dirname(ptxUri.fsPath);
  const stem = path.basename(ptxUri.fsPath, path.extname(ptxUri.fsPath));
  const stemNoHash = stripHashSuffix(stem);

  const entries = await vscode.workspace.fs.readDirectory(vscode.Uri.file(dir));
  const sassFiles = entries
    .filter(
      ([name, type]) =>
        type === vscode.FileType.File && isSassLikeExt(path.extname(name))
    )
    .map(([name]) => ({
      name,
      uri: vscode.Uri.file(path.join(dir, name)),
      stem: path.basename(name, path.extname(name)),
      ext: path.extname(name).toLowerCase(),
    }));

  const exact = sassFiles.find((f) => f.stem.toLowerCase() === stem.toLowerCase());
  if (exact) {
    return exact.uri;
  }

  const hashTrimmed = sassFiles.find(
    (f) => f.stem.toLowerCase() === stemNoHash.toLowerCase()
  );
  if (hashTrimmed) {
    return hashTrimmed.uri;
  }

  if (sassFiles.length === 1) {
    return sassFiles[0]!.uri;
  }

  return undefined;
}
