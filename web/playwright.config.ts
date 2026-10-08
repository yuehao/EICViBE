import { defineConfig } from "@playwright/test";

// Browser tests against `eicvibe-web --demo`, using the installed Chrome
// (PW_CHANNEL=chromium uses Playwright's own browser after `npx playwright install`).
const port = 8799;

export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  fullyParallel: true,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${port}/`,
    channel: process.env.PW_CHANNEL ?? "chrome",
    viewport: { width: 1440, height: 900 },
  },
  webServer: {
    command: `npm run build && uv run --project .. eicvibe-web --demo --port ${port}`,
    url: `http://127.0.0.1:${port}/api/v1/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
