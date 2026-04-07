import { Worker } from "worker_threads";
import * as path from "path";

export function runWorkerScript(
  scriptRelativeToAnalyzerDir: string,
  data: unknown
): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, scriptRelativeToAnalyzerDir);
    const w = new Worker(scriptPath, { workerData: data });
    let settled = false;
    w.on("message", (msg) => {
      settled = true;
      resolve(msg);
    });
    w.on("error", (err) => {
      settled = true;
      reject(err);
    });
    w.on("exit", (code) => {
      if (!settled && code !== 0) {
        reject(new Error(`Worker exited with code ${code}`));
      }
    });
  });
}
