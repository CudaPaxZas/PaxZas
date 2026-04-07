import path from "path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    pool: "forks",
    maxWorkers: 1,
  },
  resolve: {
    alias: {
      "@src": path.resolve(process.cwd(), "src"),
    },
  },
});
