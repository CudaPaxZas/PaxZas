import { parentPort, workerData } from "worker_threads";
import { analyzePattern } from "../pattern_model";
import type { PtxInstructionFeatures } from "../ptx_features";
import type { SassInstructionFeatures } from "../sass_features";

type PatPayload = {
  instr: PtxInstructionFeatures;
  sassInstr: SassInstructionFeatures | null;
};

const d = workerData as PatPayload;
parentPort!.postMessage(
  analyzePattern(d.instr, d.sassInstr ?? undefined)
);
