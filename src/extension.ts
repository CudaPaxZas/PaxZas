import * as vscode from "vscode";
import { analyze, type AnalyzerReport } from "./analyzer/analyze";
import {
  findCudaArtifactUris,
  findSupplementalSassForPtx,
} from "./cuda_artifacts";
import {
  allPresetMetadata,
  presetForSmVersion,
  resolveGpuSpecForAnalysis,
} from "./analyzer/gpu_spec";
import { mergeLaunchWithHints } from "./analyzer/merge_launch";
import { detectSassSmTargets, extractSassFeatures } from "./analyzer/sass_features";
import { sweepBlockSizes, buildSignals, type KernelInsights } from "./analyzer/occupancy_sweep";
import { showKernelDiagnosisPanel, buildDiagnosisPayload } from "./webview/diagnosisPanel";

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

async function supplementalSassForPtx(
  sourceUri: vscode.Uri | undefined
): Promise<string | undefined> {
  if (!sourceUri || sourceUri.scheme !== "file") {
    return undefined;
  }
  const ext = sourceUri.fsPath.toLowerCase();
  if (!ext.endsWith(".ptx")) {
    return undefined;
  }
  const sassUri = await findSupplementalSassForPtx(sourceUri);
  if (!sassUri) {
    return undefined;
  }
  return readFileText(sassUri);
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

function looksLikeSassDump(text: string): boolean {
  return /^\s*Function\s*:/m.test(text);
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
  if (r.analysis_mode) {
    lines.push(`Analysis mode: ${r.analysis_mode}`);
  }
  if (r.analysis_mode_note) {
    lines.push(r.analysis_mode_note);
  }
  if (r.sass_detected_targets?.length) {
    lines.push(`Detected SASS targets: ${r.sass_detected_targets.join(", ")}`);
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
  launch?: Partial<Record<"threads" | "shared" | "registers" | "grid", number>>,
  supplementalSass?: string
): Promise<void> {
  const t0 = Date.now();
  const result = await analyze(
    text,
    launch ?? {},
    undefined,
    supplementalSass,
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
        const supplementalSass = await supplementalSassForPtx(uri);
        await runAnalysisOnText(text, uri.fsPath, undefined, supplementalSass);
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
        const supplementalSass = await supplementalSassForPtx(doc.uri);
        await runAnalysisOnText(
          text,
          doc.uri.fsPath || doc.fileName,
          undefined,
          supplementalSass
        );
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
      const supplementalSass = await supplementalSassForPtx(picked!);
      await runAnalysisOnText(
        text,
        picked!.fsPath,
        undefined,
        supplementalSass
      );
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
      "paxzas.kernelDiagnosis",
      async (uri?: vscode.Uri) => {
        await runKernelDiagnosisCommand(context, uri);
      }
    )
  );


  context.subscriptions.push(
    vscode.commands.registerCommand(
      "paxzas.analyzeWithLaunchSpec",
      async (uri?: vscode.Uri) => {
        const spec = await vscode.window.showInputBox({
          prompt:
            "Optional launch: threads=N,shared=N,regs=N,grid=N (comma-separated)",
          placeHolder: "threads=128,shared=0,regs=32,grid=1024",
          ignoreFocusOut: true,
        });
        if (spec === undefined) {
          return;
        }
        const launch = parseLaunchSpec(spec);
        let pathLabel = "";
        let text: string;
        let supplementalSass: string | undefined;
        if (uri) {
          text = await readFileText(uri);
          pathLabel = uri.fsPath;
          supplementalSass = await supplementalSassForPtx(uri);
        } else {
          const doc = vscode.window.activeTextEditor?.document;
          if (!doc) {
            return;
          }
          pathLabel = doc.uri.fsPath || doc.fileName;
          text = await textFromDocument(doc);
          supplementalSass = await supplementalSassForPtx(doc.uri);
        }
        const t0 = Date.now();
        const r = await analyze(
          text,
          launch,
          undefined,
          supplementalSass,
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


/** Flatten SassInstructionFeatures to a plain Record<string, number> for webview transfer. */
function flattenSassFeatures(
  sass: ReturnType<typeof extractSassFeatures>[1]
): Record<string, number> {
  const out: Record<string, number> = {};
  const skip = new Set(["instruction_sequence"]);
  for (const [k, v] of Object.entries(sass)) {
    if (!skip.has(k) && typeof v === "number") {
      out[k] = v;
    }
  }
  return out;
}

/** Shared source-text resolution used by the new panel commands. */
async function resolveSourceText(
  uri: vscode.Uri | undefined,
  placeHolder: string
): Promise<{ text: string; sourceUri: vscode.Uri } | undefined> {
  if (uri) {
    return { text: await readFileText(uri), sourceUri: uri };
  }
  const editor = vscode.window.activeTextEditor;
  if (editor && isSupportedDocument(editor.document)) {
    return {
      text: await textFromDocument(editor.document),
      sourceUri: editor.document.uri,
    };
  }
  const uris = await findCudaArtifactUris();
  if (uris.length === 0) {
    void vscode.window.showWarningMessage("No .ptx or .sass files found.");
    return undefined;
  }
  let picked: vscode.Uri;
  if (uris.length === 1) {
    picked = uris[0]!;
  } else {
    const items = uris.map((u) => ({
      label: vscode.workspace.asRelativePath(u, false),
      description: u.fsPath,
      uri: u,
    }));
    const sel = await vscode.window.showQuickPick(items, {
      placeHolder,
      matchOnDescription: true,
    });
    if (!sel) return undefined;
    picked = sel.uri;
  }
  return { text: await readFileText(picked), sourceUri: picked };
}

/** Extract occupancy/diagnosis/sweep from a per-preset report into the payload shape. */
function capabilityResultFromReport(
  meta: { key: string; label: string; smTag: string },
  report: AnalyzerReport,
  detectedTargetSet: Set<string>,
  shared: number,
  registers: number,
  currentBlockSize: number | undefined,
  specForPreset: import("./analyzer/gpu_spec").GpuSpec
): NonNullable<import("./webview/diagnosisPanel").DiagnosisPayload["capabilityResults"]>[number] | undefined {
  if (report.error || !report.memory || !report.pattern) return undefined;
  const occ = report.kernel;
  const occModel = report.occupancy_model;
  const diag = report.diagnosis;
  // Sweep is recomputed per preset because occupancy curves depend on SM limits
  // (threads/registers/shared/block limits), not just instruction features.
  const sweep = sweepBlockSizes(shared, registers, specForPreset, currentBlockSize, undefined);
  return {
    preset: meta.key,
    label: meta.label,
    smTag: meta.smTag,
    isNative: detectedTargetSet.has(meta.smTag),
    analysisMode: report.analysis_mode ?? "preset-only-what-if",
    gpu: specForPreset.name,
    gpuSpec: {
      smMaxThreads: specForPreset.smMaxThreads,
      smMaxWarps: specForPreset.smMaxWarps,
      smMaxRegisters: specForPreset.smMaxRegisters,
      smMaxSharedMem: specForPreset.smMaxSharedMem,
      smMaxBlocks: specForPreset.smMaxBlocks,
      smCount: specForPreset.smCount,
      warpSize: specForPreset.warpSize,
    },
    occupancy: {
      class: occModel?.class ?? "unknown",
      confidence: occModel?.confidence ?? 0,
      occupancy: (occ?.occupancy as number) ?? 0,
      limiting_factor: occModel?.limiting_factor ?? "unknown",
      blocks_per_sm: occ?.blocks_per_sm ?? 0,
      threads_per_block: occModel?.threads_per_block ?? 0,
      shared_mem_per_block: occModel?.shared_mem_per_block ?? 0,
      registers_per_thread: occModel?.registers_per_thread ?? 0,
      occupancy_class: occ?.occupancy_class ?? "unknown",
      register_pressure_margin: occ?.register_pressure_margin,
      next_occupancy_class: occ?.next_occupancy_class,
      estimated_sm_utilization: occ?.estimated_sm_utilization,
      warnings: occ?.warnings ?? [],
      limits: occ?.limits ?? {},
      waste_metrics: (occ?.waste_metrics as Record<string, number>) ?? {},
      sources: occModel?.sources ?? { threads: "unknown", shared: "unknown", registers: "unknown" },
    },
    diagnosis: {
      primary_bottleneck: diag?.primary_bottleneck ?? "none_detected",
      secondary_bottlenecks: diag?.secondary_bottlenecks ?? [],
      stall_profile: {
        memory_dependency: diag?.stall_profile.memory_dependency ?? false,
        memory_throttle: diag?.stall_profile.memory_throttle ?? false,
        local_memory: diag?.stall_profile.local_memory ?? false,
        sync: diag?.stall_profile.sync ?? false,
      },
      optimization_priority: diag?.optimization_priority ?? [],
      confidence: diag?.confidence ?? 0,
    },
    confidences: {
      occupancy: occModel?.confidence ?? 0,
      diagnosis: diag?.confidence ?? 0,
    },
    sweep: {
      registersPerThread: registers,
      sharedMemPerBlock: shared,
      currentBlockSize,
      points: sweep.points,
    },
  };
}

async function runKernelDiagnosisCommand(
  context: vscode.ExtensionContext,
  uri?: vscode.Uri
): Promise<void> {
  const src = await resolveSourceText(uri, "Select PTX/SASS file for kernel analysis");
  if (!src) return;
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Kernel Analysis", cancellable: false },
    async () => {
      await new Promise<void>((r) => setImmediate(r));

      // ── 1. Detect SASS targets ──────────────────────────────────────────
      const sassText = await supplementalSassForPtx(src.sourceUri);
      const embeddedSass = looksLikeSassDump(src.text) ? src.text : undefined;
      const effectiveSassText = sassText ?? embeddedSass;
      const detectedSmTargets = effectiveSassText ? detectSassSmTargets(effectiveSassText) : [];
      const detectedTargetTags = detectedSmTargets.map((sm) => `sm_${sm}`);
      const detectedTargetSet = new Set(detectedTargetTags);

      // ── 2. Determine default capability: first native target, else settings ──
      const settingsPreset = gpuPresetFromSettings();
      let defaultPreset = settingsPreset;
      if (settingsPreset === "auto" || !allPresetMetadata().find((p) => p.key === settingsPreset)) {
        const firstNative = detectedSmTargets.map((sm) => presetForSmVersion(sm)).find((p) => p != null);
        if (firstNative) defaultPreset = firstNative.key;
        else defaultPreset = "a100";
      }

      // ── 3. SASS features (shared across all presets) ───────────────────
      // Use the sidecar file first; fall back to the file itself if it is a SASS dump opened directly.
      const sassSource = sassText ?? embeddedSass;
      let sassFlatFeatures: Record<string, number> | undefined;
      let sassRegisters: number | undefined;
      if (sassSource) {
        const [, sassInstr] = extractSassFeatures(sassSource, undefined);
        sassFlatFeatures = flattenSassFeatures(sassInstr);
        if (sassInstr.max_register_index >= 0) {
          sassRegisters = sassInstr.max_register_index + 1;
        }
      }

      // ── 4. Kernel launch params (shared across all presets) ───────────
      let registers = 32;
      let shared = 0;
      let currentBlockSize: number | undefined;
      try {
        const merged = mergeLaunchWithHints(src.text, undefined, {}, sassRegisters ?? null);
        registers = merged.registers;
        shared = merged.shared;
        currentBlockSize = merged.threads;
      } catch {
        if (sassRegisters != null) registers = sassRegisters;
      }

      // ── 5. Run analysis for ALL presets in parallel ────────────────────
      const allMeta = allPresetMetadata();
      const [allReports, allSpecs] = await Promise.all([
        // Instruction-level analysis for every preset.
        Promise.all(allMeta.map((meta) => analyze(src.text, {}, undefined, sassText, meta.key))),
        // Resolve each preset's concrete SM limits used by occupancy/sweep.
        Promise.all(allMeta.map((meta) =>
          resolveGpuSpecForAnalysis(meta.key, { ptxText: src.text, enableLocalCudaDetect: false })
            .then((r) => r.spec)
        )),
      ]);

      // ── 6. Primary report (for top-level payload) ───────────────────────
      const defaultIdx = allMeta.findIndex((m) => m.key === defaultPreset);
      const primaryIdx = defaultIdx >= 0 ? defaultIdx : 0;
      const report = allReports[primaryIdx]!;
      const spec = allSpecs[primaryIdx]!;
      if (report.error) {
        void vscode.window.showErrorMessage(`Kernel Analysis failed: ${report.error}`);
        return;
      }

      // ── 7. Build sweepInsights for primary (pattern/memory are instruction-driven) ──
      let sweepInsights: KernelInsights | undefined;
      if (report.memory && report.pattern) {
        const signals = buildSignals(report.pattern, report.memory);
        sweepInsights = {
          memoryClass: report.memory.class,
          memoryInsight: report.memory.insight,
          memoryConfidence: report.memory.confidence,
          arithmeticIntensity: report.memory.arithmetic_intensity_ops_per_byte,
          cachePolicy: report.memory.cache_policy,
          reuseRatio: report.memory.reuse_ratio,
          memComputeRatio: report.memory.mem_compute_ratio,
          loadStoreRatio: report.memory.load_store_ratio,
          patternClass: report.pattern.class,
          patternInsight: report.pattern.insight,
          patternConfidence: report.pattern.confidence,
          archetype: report.pattern.archetype,
          archetypeDescriptors: [],
          primaryBottleneck: report.diagnosis?.primary_bottleneck ?? "none_detected",
          secondaryBottlenecks: report.diagnosis?.secondary_bottlenecks ?? [],
          suggestions: report.diagnosis?.optimization_priority?.slice(0, 3) ?? [],
          signals,
          occupancyConfidence: report.occupancy_model?.confidence ?? 0,
          diagnosisConfidence: report.diagnosis?.confidence ?? 0,
          stallProfile: {
            memoryDependency: report.diagnosis?.stall_profile.memory_dependency ?? false,
            memoryThrottle: report.diagnosis?.stall_profile.memory_throttle ?? false,
            localMemory: report.diagnosis?.stall_profile.local_memory ?? false,
            sync: report.diagnosis?.stall_profile.sync ?? false,
          },
          launchWarnings: report.kernel?.warnings ?? [],
          sources: report.occupancy_model?.sources ?? { threads: "unknown", shared: "unknown", registers: "unknown" },
          registerPressureMargin: report.kernel?.register_pressure_margin,
          nextOccupancyClass: report.kernel?.next_occupancy_class,
          estimatedSmUtilization: report.kernel?.estimated_sm_utilization,
          wasteMetrics: report.kernel ? {
            unusedThreadsPerSm: report.kernel.waste_metrics.unused_threads_per_sm,
            unusedWarpsPerSm: report.kernel.waste_metrics.unused_warps_per_sm,
            unusedRegistersPerSm: report.kernel.waste_metrics.unused_registers_per_sm,
            unusedSharedMemBytesPerSm: report.kernel.waste_metrics.unused_shared_mem_bytes_per_sm,
          } : undefined,
          vectorizationScore: report.pattern.vectorization_score,
          storeVectorizationScore: report.memory.store_vectorization_score,
          storeUncoalescedRisk: report.pattern.store_uncoalesced_risk,
          ptxOnly: !sassText,
        };
      }
      const primarySweep = sweepBlockSizes(shared, registers, spec, currentBlockSize, sweepInsights);

      // ── 8. Build capabilityResults for all presets ─────────────────────
      const capabilityResults = allMeta
        .map((meta, i) => capabilityResultFromReport(
          meta, allReports[i]!, detectedTargetSet, shared, registers, currentBlockSize, allSpecs[i]!
        ))
        .filter((r): r is NonNullable<typeof r> => r != null);
      // Keep a fixed preset list in UI order. Native-vs-what-if is encoded per row.

      // ── 9. Build and show payload ───────────────────────────────────────
      const payload = buildDiagnosisPayload(
        src.sourceUri.fsPath,
        report,
        spec,
        sassFlatFeatures,
        { registersPerThread: registers, sharedMemPerBlock: shared, currentBlockSize, points: primarySweep.points },
        capabilityResults,
        defaultPreset
      );
      if (!payload) {
        void vscode.window.showWarningMessage("Kernel Analysis: insufficient data from analysis.");
        return;
      }
      showKernelDiagnosisPanel(context.extensionUri, payload);
    }
  );
}


function parseLaunchSpec(
  raw: string
): Partial<Record<"threads" | "shared" | "registers" | "grid", number>> {
  const out: Partial<Record<"threads" | "shared" | "registers" | "grid", number>> =
    {};
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
    } else if (k === "grid" || k === "blocks" || k === "g") {
      out.grid = v;
    }
  }
  return out;
}

export function deactivate(): void {}
