import * as vscode from "vscode";
import { analyze, type AnalyzerReport } from "./analyzer/analyze";
import { findCudaArtifactUris } from "./cuda_artifacts";

const OUTPUT_CHANNEL_ID = "cudaAnalyzer";

function isSupportedDocument(doc: vscode.TextDocument): boolean {
  const ext = doc.fileName.toLowerCase();
  return (
    doc.languageId === "cuda" ||
    ext.endsWith(".ptx") ||
    ext.endsWith(".sass") ||
    ext.endsWith(".s") ||
    ext.endsWith(".cu")
  );
}

async function readFileText(uri: vscode.Uri): Promise<string> {
  const bytes = await vscode.workspace.fs.readFile(uri);
  return new TextDecoder("utf-8").decode(bytes);
}

/** Prefer disk read for clean `file:` docs (single decode); buffer if dirty or virtual. */
async function textFromDocument(doc: vscode.TextDocument): Promise<string> {
  if (doc.isDirty || doc.uri.scheme !== "file") {
    return doc.getText();
  }
  return readFileText(doc.uri);
}

function gpuPresetFromSettings(): string {
  return (
    vscode.workspace.getConfiguration("paxzas").get<string>("gpuPreset") ??
    "auto"
  );
}

function formatSummary(r: AnalyzerReport, meta?: { path?: string; ms?: number }): string {
  const head: string[] = [];
  if (meta?.path) {
    head.push(`File: ${meta.path}`);
  }
  if (meta?.ms !== undefined) {
    head.push(`Analysis time: ${meta.ms} ms`);
  }
  if (r.gpu_spec_resolution) {
    head.push(`GPU spec: ${r.gpu_spec_resolution}`);
  }
  if (head.length) {
    head.push("");
  }

  if (r.error) {
    return head.join("\n") + `Error: ${r.error}`;
  }
  const lines: string[] = [...head];
  if (r.sass_note) {
    lines.push(r.sass_note);
    lines.push("");
  }
  lines.push(`PTX kernel (launch/occupancy): ${r.ptx_kernel ?? "(none)"}`);
  if (r.ptx_kernels_analyzed !== undefined && r.ptx_kernels_analyzed > 0) {
    lines.push(`PTX kernels scanned for features: ${r.ptx_kernels_analyzed}`);
  }
  if (r.ptx_multi_kernel_note) {
    lines.push(r.ptx_multi_kernel_note);
    lines.push("");
  }
  if (r.ptx_hints) {
    lines.push(
      `Hints: maxnreg=${r.ptx_hints.maxnreg ?? "?"} maxntid=${JSON.stringify(r.ptx_hints.maxntid)} static_shared=${r.ptx_hints.static_shared_bytes_est} dynamic_shared=${r.ptx_hints.dynamic_shared_detected}`
    );
  }
  if (r.ptx_features) {
    lines.push(`PTX features: ${JSON.stringify(r.ptx_features)}`);
  }
  if (r.pattern) {
    lines.push(
      `Pattern: ${r.pattern.class} (confidence ${r.pattern.confidence}, source ${r.pattern.source})`
    );
    lines.push(`  ${r.pattern.insight}`);
  }
  if (r.memory) {
    lines.push(
      `Memory: ${r.memory.class} (confidence ${r.memory.confidence}) — ${r.memory.insight}`
    );
  }
  if (r.bottleneck_heuristic) {
    lines.push(
      `Bottleneck: ${r.bottleneck_heuristic.bottleneck} intensity=${r.bottleneck_heuristic.arithmetic_intensity_ops_per_byte ?? "inf"}`
    );
  }
  if (r.occupancy_model) {
    lines.push(
      `Occupancy model: ${r.occupancy_model.class} occ=${r.occupancy_model.occupancy} limit=${r.occupancy_model.limiting_factor}`
    );
  }
  if (r.kernel) {
    lines.push(
      `Kernel SM: blocks/SM=${r.kernel.blocks_per_sm} warps/SM=${r.kernel.warp_metrics.warps_per_sm}`
    );
  }
  if (r.register_estimates) {
    lines.push(`Registers: ${JSON.stringify(r.register_estimates)} (${r.register_source})`);
  }
  return lines.join("\n");
}

export async function runAnalysisOnText(
  text: string,
  displayPath: string,
  launch?: Partial<Record<"threads" | "shared" | "registers", number>>
): Promise<void> {
  const t0 = Date.now();
  const result = await analyze(
    text,
    launch ?? {},
    undefined,
    undefined,
    gpuPresetFromSettings()
  );
  const ms = Date.now() - t0;

  const ch = vscode.window.createOutputChannel(OUTPUT_CHANNEL_ID);
  ch.clear();
  ch.appendLine(formatSummary(result, { path: displayPath, ms }));
  ch.show(true);

  if (result.error) {
    void vscode.window.showErrorMessage(result.error);
    return;
  }
  const pat = result.pattern?.class ?? "?";
  void vscode.window.showInformationMessage(
    `CUDA: pattern=${pat} — ${ms} ms — see Output (CUDA Analyzer)`
  );
}

async function runCudaAnalyzeCommand(uri?: vscode.Uri): Promise<void> {
  if (uri) {
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "CUDA Analyzer",
        cancellable: false,
      },
      async () => {
        await new Promise<void>((r) => setImmediate(r));
        const text = await readFileText(uri);
        await runAnalysisOnText(text, uri.fsPath);
      }
    );
    return;
  }

  const editor = vscode.window.activeTextEditor;
  if (editor && isSupportedDocument(editor.document)) {
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "CUDA Analyzer",
        cancellable: false,
      },
      async () => {
        await new Promise<void>((r) => setImmediate(r));
        const doc = editor.document;
        const text = await textFromDocument(doc);
        await runAnalysisOnText(text, doc.uri.fsPath || doc.fileName);
      }
    );
    return;
  }

  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) {
    void vscode.window.showErrorMessage(
      "Open a workspace, or a .ptx / .sass file, or use Explorer context menu."
    );
    return;
  }

  const uris = await findCudaArtifactUris();
  if (uris.length === 0) {
    void vscode.window.showWarningMessage(
      "No .ptx or .sass files found in this workspace."
    );
    return;
  }

  let picked: vscode.Uri | undefined;
  if (uris.length === 1) {
    picked = uris[0];
  } else {
    const items = uris.map((u) => ({
      label: vscode.workspace.asRelativePath(u, false),
      description: u.fsPath,
      uri: u,
    }));
    const sel = await vscode.window.showQuickPick(items, {
      placeHolder: "Select PTX/SASS file to analyze",
      matchOnDescription: true,
    });
    picked = sel?.uri;
  }
  if (!picked) {
    return;
  }

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "CUDA Analyzer",
      cancellable: false,
    },
    async () => {
      await new Promise<void>((r) => setImmediate(r));
      const text = await readFileText(picked!);
      await runAnalysisOnText(text, picked!.fsPath);
    }
  );
}

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("cudaAnalyzer.run", () => runCudaAnalyzeCommand())
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("paxzas.analyzeActiveFile", () =>
      runCudaAnalyzeCommand()
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("paxzas.analyzeSelectedFile", (u: vscode.Uri) =>
      runCudaAnalyzeCommand(u)
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "paxzas.analyzeWithLaunchSpec",
      async (uri?: vscode.Uri) => {
        const spec = await vscode.window.showInputBox({
          prompt:
            "Optional launch: threads=N,shared=N,regs=N (comma-separated)",
          placeHolder: "threads=128,shared=0,regs=32",
          ignoreFocusOut: true,
        });
        if (spec === undefined) {
          return;
        }
        const launch = parseLaunchSpec(spec);
        let pathLabel = "";
        let text: string;
        if (uri) {
          text = await readFileText(uri);
          pathLabel = uri.fsPath;
        } else {
          const doc = vscode.window.activeTextEditor?.document;
          if (!doc) {
            return;
          }
          pathLabel = doc.uri.fsPath || doc.fileName;
          text = await textFromDocument(doc);
        }
        const t0 = Date.now();
        const r = await analyze(
          text,
          launch,
          undefined,
          undefined,
          gpuPresetFromSettings()
        );
        const ms = Date.now() - t0;
        const ch = vscode.window.createOutputChannel(OUTPUT_CHANNEL_ID);
        ch.clear();
        ch.appendLine(formatSummary(r, { path: pathLabel, ms }));
        ch.show(true);
        if (r.error) {
          void vscode.window.showErrorMessage(r.error);
        } else {
          void vscode.window.showInformationMessage(
            `CUDA: pattern=${r.pattern?.class ?? "?"} (launch hints applied, ${ms} ms)`
          );
        }
      }
    )
  );
}

function parseLaunchSpec(
  raw: string
): Partial<Record<"threads" | "shared" | "registers", number>> {
  const out: Partial<Record<"threads" | "shared" | "registers", number>> = {};
  const parts = raw.split(/[,;]/);
  for (const part of parts) {
    const m = part.trim().match(/^(\w+)\s*=\s*(\d+)/i);
    if (!m) {
      continue;
    }
    const k = m[1]!.toLowerCase();
    const v = parseInt(m[2]!, 10);
    if (k === "threads" || k === "t") {
      out.threads = v;
    } else if (k === "shared" || k === "s" || k === "shmem") {
      out.shared = v;
    } else if (k === "regs" || k === "registers" || k === "r") {
      out.registers = v;
    }
  }
  return out;
}

export function deactivate(): void {}
