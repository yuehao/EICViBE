import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The built client is served by the Python package (eicvibe.api); `npm run dev`
// serves it with hot reload and forwards API calls to `eicvibe-web` on :8000.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../src/eicvibe/api/static",
    emptyOutDir: true,
    // The lattice viewer, the virtual accelerator's admin page (ADR 0008) and the tracking check.
    rollupOptions: { input: { main: "index.html", admin: "admin.html", check: "check.html" } },
  },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { include: ["src/**/*.test.ts"] },
});
