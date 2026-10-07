import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The built client is served by the Python package (eicvibe.api); `npm run dev`
// serves it with hot reload and forwards API calls to `eicvibe-web` on :8000.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../src/eicvibe/api/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { include: ["src/**/*.test.ts"] },
});
