import { parentPort, workerData } from "worker_threads";
import { extractOneKernelFeatures } from "../ptx_features";

type Payload = { ptx: string; kernelSubstring: string };

const d = workerData as Payload;
parentPort!.postMessage(extractOneKernelFeatures(d.ptx, d.kernelSubstring));
