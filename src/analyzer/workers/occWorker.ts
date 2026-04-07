import { parentPort, workerData } from "worker_threads";
import { analyzeOccupancyModel } from "../occupancy_model";
import type { GpuSpec } from "../gpu_spec";

type OccPayload = {
  threads: number;
  shared: number;
  registers: number;
  spec: GpuSpec;
  threadsSource: string;
  sharedSource: string;
  registerSource: string;
};

const d = workerData as OccPayload;
parentPort!.postMessage(
  analyzeOccupancyModel(
    d.threads,
    d.shared,
    d.registers,
    d.spec,
    d.threadsSource,
    d.sharedSource,
    d.registerSource
  )
);
