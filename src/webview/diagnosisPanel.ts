import * as vscode from "vscode";
import * as crypto from "crypto";
import type { AnalyzerReport } from "../analyzer/analyze";
import type { GpuSpec } from "../analyzer/gpu_spec";

export interface DiagnosisPayload {
  displayPath: string;
  kernelName: string | undefined;
  gpu: string;
  gpuSpec: {
    smMaxThreads: number;
    smMaxWarps: number | undefined;
    smMaxRegisters: number;
    smMaxSharedMem: number;
    smMaxBlocks: number;
    smCount: number | undefined;
    warpSize: number;
  };
  confidences: { memory: number; pattern: number; occupancy: number; diagnosis: number };
  diagnosis: {
    primary_bottleneck: string;
    secondary_bottlenecks: string[];
    stall_profile: {
      memory_dependency: boolean;
      memory_throttle: boolean;
      local_memory: boolean;
      sync: boolean;
    };
    optimization_priority: string[];
    confidence: number;
  };
  memory: {
    class: string;
    insight: string;
    global_loads: number;
    global_stores: number;
    shared_loads: number;
    shared_stores: number;
    memory_pressure: number;
    cache_policy: string | null;
    confidence: number;
    arithmetic_intensity_ops_per_byte: number | null;
    reuse_ratio: number;
    reuse_strength: number;
    mem_compute_ratio: number;
    store_vectorization_score: number;
    load_store_ratio: number;
    load_store_balance: string;
    bytes_proxy: number;
    flops_proxy: number;
    global_mem_source: string;
  };
  pattern: {
    class: string;
    insight: string;
    archetype: string | undefined;
    confidence: number;
    source: string;
    high_looping: boolean;
    sync_heavy: boolean;
    tensor_dominated: boolean;
    streaming: boolean;
    sync_efficiency: string;
    interleaving: string;
    spill_risk: boolean;
    spill_severity: number;
    uncoalesced_risk: boolean;
    store_uncoalesced_risk: boolean;
    missing_tensor_cores: boolean;
    atomic_contention_risk: boolean;
    sfu_heavy: boolean;
    vectorization_score: number;
    store_vectorization_score: number;
    over_synchronized: boolean;
    fp16_scalar_risk: boolean;
    read_modify_write: boolean;
    tensor_utilization_fraction: number;
    productive_instruction_fraction: number;
    warp_divergence_risk: boolean;
    uses_warp_shuffle: boolean;
    uses_warp_vote: boolean;
    warp_reduction_pattern: boolean;
    compute_to_memory_ratio: number;
    shared_to_global_ratio: number;
    uses_tensor_cores: boolean;
  };
  occupancy: {
    class: string;
    confidence: number;
    occupancy: number;
    limiting_factor: string;
    blocks_per_sm: number;
    threads_per_block: number;
    shared_mem_per_block: number;
    registers_per_thread: number;
    occupancy_class: string;
    register_pressure_margin: number | undefined;
    next_occupancy_class: string | undefined;
    estimated_sm_utilization: string | undefined;
    warnings: string[];
    limits: Record<string, number>;
    waste_metrics: Record<string, number>;
    sources: { threads: string; shared: string; registers: string };
  };
  ptxFeatures: Record<string, number>;
  sassFeatures: Record<string, number> | undefined;
  hasSass: boolean;
  sweep: {
    registersPerThread: number;
    sharedMemPerBlock: number;
    currentBlockSize: number | undefined;
    points: Array<{
      blockSize: number;
      occupancy: number;
      blocksPerSm: number;
      limitingFactor: string;
      blocksByThreads: number;
      blocksByWarps: number;
      blocksByShared: number;
      blocksByRegisters: number;
      blocksByBlockLimit: number;
    }>;
  } | undefined;
}

export function buildDiagnosisPayload(
  displayPath: string,
  report: AnalyzerReport,
  spec: GpuSpec,
  sassFeatures: Record<string, number> | undefined,
  sweep?: DiagnosisPayload["sweep"]
): DiagnosisPayload | undefined {
  if (report.error || !report.memory || !report.pattern) {
    return undefined;
  }
  const mem = report.memory;
  const pat = report.pattern;
  const occ = report.kernel;
  const diag = report.diagnosis;
  const occModel = report.occupancy_model;

  return {
    displayPath,
    kernelName: report.ptx_kernel,
    gpu: spec.name,
    gpuSpec: {
      smMaxThreads: spec.smMaxThreads,
      smMaxWarps: spec.smMaxWarps,
      smMaxRegisters: spec.smMaxRegisters,
      smMaxSharedMem: spec.smMaxSharedMem,
      smMaxBlocks: spec.smMaxBlocks,
      smCount: spec.smCount,
      warpSize: spec.warpSize,
    },
    confidences: {
      memory: mem.confidence,
      pattern: pat.confidence,
      occupancy: occModel?.confidence ?? 0,
      diagnosis: diag?.confidence ?? 0,
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
    memory: {
      class: mem.class,
      insight: mem.insight,
      global_loads: mem.global_loads,
      global_stores: mem.global_stores,
      shared_loads: mem.shared_loads,
      shared_stores: mem.shared_stores,
      memory_pressure: mem.memory_pressure,
      cache_policy: mem.cache_policy,
      confidence: mem.confidence,
      arithmetic_intensity_ops_per_byte: mem.arithmetic_intensity_ops_per_byte,
      reuse_ratio: mem.reuse_ratio,
      reuse_strength: mem.reuse_strength,
      mem_compute_ratio: mem.mem_compute_ratio,
      store_vectorization_score: mem.store_vectorization_score,
      load_store_ratio: mem.load_store_ratio,
      load_store_balance: mem.load_store_balance,
      bytes_proxy: mem.bytes_proxy,
      flops_proxy: mem.flops_proxy,
      global_mem_source: mem.global_mem_source,
    },
    pattern: {
      class: pat.class,
      insight: pat.insight,
      archetype: pat.archetype,
      confidence: pat.confidence,
      source: pat.source,
      high_looping: pat.high_looping,
      sync_heavy: pat.sync_heavy,
      tensor_dominated: pat.tensor_dominated,
      streaming: pat.streaming,
      sync_efficiency: pat.sync_efficiency,
      interleaving: pat.interleaving,
      spill_risk: pat.spill_risk,
      spill_severity: pat.spill_severity,
      uncoalesced_risk: pat.uncoalesced_risk,
      store_uncoalesced_risk: pat.store_uncoalesced_risk,
      missing_tensor_cores: pat.missing_tensor_cores,
      atomic_contention_risk: pat.atomic_contention_risk,
      sfu_heavy: pat.sfu_heavy,
      vectorization_score: pat.vectorization_score,
      store_vectorization_score: pat.store_vectorization_score,
      over_synchronized: pat.over_synchronized,
      fp16_scalar_risk: pat.fp16_scalar_risk,
      read_modify_write: pat.read_modify_write,
      tensor_utilization_fraction: pat.tensor_utilization_fraction,
      productive_instruction_fraction: pat.productive_instruction_fraction,
      warp_divergence_risk: pat.warp_divergence_risk,
      uses_warp_shuffle: pat.uses_warp_shuffle,
      uses_warp_vote: pat.uses_warp_vote,
      warp_reduction_pattern: pat.warp_reduction_pattern,
      compute_to_memory_ratio: pat.compute_to_memory_ratio,
      shared_to_global_ratio: pat.shared_to_global_ratio,
      uses_tensor_cores: pat.uses_tensor_cores,
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
      sources: occModel?.sources ?? {
        threads: "unknown",
        shared: "unknown",
        registers: "unknown",
      },
    },
    ptxFeatures: report.ptx_features ?? {},
    sassFeatures,
    hasSass: sassFeatures != null,
    sweep,
  };
}

let currentPanel: vscode.WebviewPanel | undefined;

export function showKernelDiagnosisPanel(
  extensionUri: vscode.Uri,
  payload: DiagnosisPayload
): void {
  const column = vscode.ViewColumn.Beside;
  if (currentPanel) {
    currentPanel.reveal(column);
    void currentPanel.webview.postMessage(payload);
    return;
  }
  const panel = vscode.window.createWebviewPanel(
    "paxzas.kernelDiagnosis",
    `Diagnosis — ${payload.kernelName ?? payload.gpu}`,
    column,
    {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(extensionUri, "media")],
      retainContextWhenHidden: true,
    }
  );
  currentPanel = panel;
  panel.onDidDispose(() => { currentPanel = undefined; });
  panel.webview.html = getHtml(panel.webview, extensionUri);
  panel.webview.onDidReceiveMessage((msg) => {
    if (msg?.type === "ready") {
      void panel.webview.postMessage(payload);
    }
  });
}

function getHtml(webview: vscode.Webview, extensionUri: vscode.Uri): string {
  const mediaUri = vscode.Uri.joinPath(extensionUri, "media");
  const chartJsUri = webview.asWebviewUri(vscode.Uri.joinPath(mediaUri, "chart.min.js"));
  const htmlUri = vscode.Uri.joinPath(mediaUri, "diagnosis.html");
  const fs = require("fs") as typeof import("fs");
  let html = fs.readFileSync(htmlUri.fsPath, "utf-8");
  const nonce = crypto.randomBytes(16).toString("base64");
  html = html.replace(/\{\{NONCE\}\}/g, nonce);
  html = html.replace(/\{\{CHART_JS_URI\}\}/g, chartJsUri.toString());
  return html;
}
