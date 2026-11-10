import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "happy-dom",
    include: ["test/**/*.test.ts"],
    // Bundle tests run esbuild, which is quick but not instant on a cold CI runner.
    testTimeout: 30_000,
    hookTimeout: 60_000,
  },
});
