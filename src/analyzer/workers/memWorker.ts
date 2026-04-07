import { parentPort, workerData } from "worker_threads";
import { analyzeMemory } from "../memory_model";
import type { PtxInstructionFeatures } from "../ptx_features";
import type { SassInstructionFeatures } from "../sass_features";

type MemPayload = {
  instr: PtxInstructionFeatures;
  sassInstr: SassInstructionFeatures | null;
};

const d = workerData as MemPayload;
parentPort!.postMessage(
  analyzeMemory(d.instr, d.sassInstr ?? undefined)
);
