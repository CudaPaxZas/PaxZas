/**
 * Local analysis entry: mirrors pipeline.analyze_ptx_text core (Phase 1–2 blocks).
 */

import { analyzeKernel } from "./occupancy_model";
import { heuristicBottleneck } from "./bottleneck";
import {
  buildFeatureBundle,
  emptyInstructionFeatures,
} from "./ptx_features";
import { extractPtxKernelsMerged } from "./parallel_kernel_extract";
import { detectSassSmTargets, extractSassFeatures } from "./sass_features";
import { mergeLaunchWithHints } from "./merge_launch";
import { getPresetMetadata, resolveGpuSpecForAnalysis } from "./gpu_spec";
import type { PtxKernelHints } from "./ptx_parse";
import { runModelsParallelOrSync } from "./run_models_parallel";
import { diagnoseKernel } from "./diagnose";
import type { DiagnosisResult } from "./diagnose";

function hasPtxEntry(ptx: string): boolean {
  return /\.(?:visible\s+)?entry\s+\S+\s*\(/.test(ptx);
}

function looksLikeSassDump(text: string): boolean {
  return /^\s*Function\s*:/m.test(text);
}

export interface AnalyzerReport {
  error?: string;
  kind: "ptx" | "sass_only";
  ptx_kernel?: string;
  ptx_hints?: {
    maxnreg: number | undefined;
    maxntid: [number, number, number] | null;
    static_shared_bytes_est: number;
    dynamic_shared_detected: boolean;
  } | null;
  ptx_features?: Record<string, number>;
  register_source?: string;
  register_estimates?: {
    ptx_maxnreg: number | undefined;
    sass_inferred: number | null | undefined;
    selected_registers_per_thread: number;
  };
  occupancy_model?: Awaited<
    ReturnType<typeof runModelsParallelOrSync>
  >["occModel"];
  memory?: Awaited<ReturnType<typeof runModelsParallelOrSync>>["memory"];
  pattern?: Awaited<ReturnType<typeof runModelsParallelOrSync>>["pattern"];
  bottleneck_heuristic?: ReturnType<typeof heuristicBottleneck>;
  kernel?: ReturnType<typeof analyzeKernel>;
  sass_note?: string;
  /** Number of `.entry` kernels used for PTX instruction features (1 or summed N). */
  ptx_kernels_analyzed?: number;
  ptx_multi_kernel_note?: string;
  /** How GpuSpec was chosen: settings preset, nvidia-smi, or default fallback. */
  gpu_spec_resolution?: string;
  /** Cross-model diagnosis: primary bottleneck, stall profile, optimisation plan. */
  diagnosis?: DiagnosisResult;
  /** SASS compute capabilities detected in the input/supplemental dump (sm_XX). */
  sass_detected_targets?: string[];
  /** Analysis alignment mode between SASS target and selected capability preset. */
  analysis_mode?: "native" | "cross-arch-what-if" | "preset-only-what-if";
  /** Human-readable detail for alignment mode. */
  analysis_mode_note?: string;
}

function parseSmFromSpecName(specName: string): number | undefined {
  const m = /sm(\d{2,3})/i.exec(specName);
  return m ? parseInt(m[1]!, 10) : undefined;
}

function parseSmVersionFromTag(smTag: string): number | undefined {
  const m = /sm_(\d{2,3})/i.exec(smTag.trim());
  return m ? parseInt(m[1]!, 10) : undefined;
}

function hintsToPublic(h: PtxKernelHints | null): AnalyzerReport["ptx_hints"] {
  if (!h) {
    return null;
  }
  return {
    maxnreg: h.maxnreg,
    maxntid: h.maxntid ? ([...h.maxntid] as [number, number, number]) : null,
    static_shared_bytes_est: h.staticSharedBytes,
    dynamic_shared_detected: h.dynamicSharedDetected,
  };
}

/**
 * @param text - file contents
 * @param launch - optional threads/shared/registers overrides (same semantics as Python CLI)
 * @param kernelSubstring - optional kernel name filter
 * @param supplementalSass - optional SASS text merged like pipeline.sass_text
 * @param gpuPreset - `auto` runs nvidia-smi once when possible; else a known preset key
 */
export async function analyze(
  text: string,
  launch: Partial<Record<"threads" | "shared" | "registers", number>> = {},
  kernelSubstring: string | undefined = undefined,
  supplementalSass: string | undefined = undefined,
  gpuPreset = "auto"
): Promise<AnalyzerReport> {
  const ptxForGpuSpec = hasPtxEntry(text) ? text : undefined;
  const { spec, resolution: gpuSpecResolution } =
    await resolveGpuSpecForAnalysis(gpuPreset, {
      ptxText: ptxForGpuSpec,
      enableLocalCudaDetect: true,
    });

  let sassKernel: string | undefined;
  let sassInstr: ReturnType<typeof extractSassFeatures>[1] | undefined;
  let sassRegisters: number | undefined;
  let sassTargets: number[] = [];

  let sassText: string | undefined = supplementalSass;
  if (!sassText && looksLikeSassDump(text) && !hasPtxEntry(text)) {
    sassText = text;
  }
  if (sassText) {
    sassTargets = detectSassSmTargets(sassText);
    [sassKernel, sassInstr] = extractSassFeatures(sassText, kernelSubstring);
    if (sassInstr.max_register_index >= 0) {
      sassRegisters = sassInstr.max_register_index + 1;
    }
  }
  const sassTargetTags = sassTargets.map((sm) => `sm_${sm}`);
  const presetMeta =
    gpuPreset.trim().toLowerCase() !== "auto" ? getPresetMetadata(gpuPreset) : undefined;
  const presetSm = presetMeta ? parseSmVersionFromTag(presetMeta.smTag) : undefined;
  const resolvedSm = parseSmFromSpecName(spec.name);
  const expectedSm = presetSm ?? resolvedSm;
  const analysisMode: AnalyzerReport["analysis_mode"] =
    sassTargets.length === 0
      ? "preset-only-what-if"
      : expectedSm !== undefined && sassTargets.includes(expectedSm)
        ? "native"
        : "cross-arch-what-if";
  const analysisModeNote =
    analysisMode === "native"
      ? `SASS targets (${sassTargetTags.join(", ")}) include resolved capability sm_${expectedSm}.`
      : analysisMode === "cross-arch-what-if"
        ? `SASS targets (${sassTargetTags.join(", ")}) do not include resolved capability sm_${expectedSm ?? "?"}; results are cross-arch what-if.`
        : `No architecture markers found in SASS; results use preset/resolved capability ${spec.name} as a what-if target.`;

  if (sassText && !hasPtxEntry(text)) {
    const instr = emptyInstructionFeatures();
    const registers = sassRegisters ?? 32;
    const threads = 256;
    const shared = 0;
    const kernel = analyzeKernel(threads, shared, registers, spec);
    const bottleneck = heuristicBottleneck(instr, registers);
    const { occModel, memory, pattern } = await runModelsParallelOrSync({
      threads,
      shared,
      registers,
      spec,
      threadsSource: "unknown",
      sharedSource: "unknown",
      registerSource:
        sassRegisters !== undefined ? "sass.inferred" : "unknown",
      instr,
      sassInstr,
    });

    return {
      kind: "sass_only",
      gpu_spec_resolution: gpuSpecResolution,
      sass_detected_targets: sassTargetTags,
      analysis_mode: analysisMode,
      analysis_mode_note: analysisModeNote,
      ptx_kernel: sassKernel,
      ptx_hints: null,
      // SASS-only: no PTX instruction counts exist — only store the resolved register count
      // so the Raw tab shows "—" for PTX instruction keys rather than misleading 0s.
      ptx_features: { registers },
      register_source:
        sassRegisters !== undefined ? "sass.inferred" : "unknown",
      register_estimates: {
        ptx_maxnreg: undefined,
        sass_inferred: sassRegisters ?? null,
        selected_registers_per_thread: registers,
      },
      occupancy_model: occModel,
      memory,
      pattern,
      bottleneck_heuristic: bottleneck,
      kernel,
      diagnosis: memory && pattern ? diagnoseKernel(memory, kernel, pattern) : undefined,
      sass_note:
        "SASS-only file: occupancy uses threads=256, shared=0 (Python pipeline normally requires PTX for launch merge).",
    };
  }

  try {
    const merged = mergeLaunchWithHints(
      text,
      kernelSubstring,
      launch,
      sassRegisters ?? null
    );
    const { instr, kernelNames } = await extractPtxKernelsMerged(
      text,
      kernelSubstring
    );
    const feat = buildFeatureBundle(merged.registers, instr);
    const multi = kernelNames.length > 1;

    const kernel = analyzeKernel(
      merged.threads,
      merged.shared,
      merged.registers,
      spec
    );
    const bottleneck = heuristicBottleneck(instr, merged.registers);
    const { occModel, memory, pattern } = await runModelsParallelOrSync({
      threads: merged.threads,
      shared: merged.shared,
      registers: merged.registers,
      spec,
      threadsSource: merged.threadsSource,
      sharedSource: merged.sharedSource,
      registerSource: merged.registerSource,
      instr,
      sassInstr,
    });

    return {
      kind: "ptx",
      gpu_spec_resolution: gpuSpecResolution,
      sass_detected_targets: sassTargetTags,
      analysis_mode: analysisMode,
      analysis_mode_note: analysisModeNote,
      ptx_kernel: merged.hints.kernelName,
      ptx_hints: hintsToPublic(merged.hints),
      ptx_features: feat,
      register_source: merged.registerSource,
      register_estimates: {
        ptx_maxnreg: merged.hints.maxnreg,
        sass_inferred: sassRegisters ?? null,
        selected_registers_per_thread: merged.registers,
      },
      occupancy_model: occModel,
      memory,
      pattern,
      bottleneck_heuristic: bottleneck,
      kernel,
      diagnosis: memory && pattern ? diagnoseKernel(memory, kernel, pattern) : undefined,
      ptx_kernels_analyzed: kernelNames.length,
      ptx_multi_kernel_note: multi
        ? `PTX instruction features are summed across ${kernelNames.length} .entry kernel(s). Pattern/memory/bottleneck use this aggregate. Occupancy/launch merge still follow the picked kernel (${merged.hints.kernelName ?? "?"}).`
        : undefined,
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return {
      error: msg,
      kind: "ptx",
      gpu_spec_resolution: gpuSpecResolution,
      sass_detected_targets: sassTargetTags,
      analysis_mode: analysisMode,
      analysis_mode_note: analysisModeNote,
      register_estimates: {
        ptx_maxnreg: undefined,
        sass_inferred: sassRegisters ?? null,
        selected_registers_per_thread: 0,
      },
    };
  }
}
