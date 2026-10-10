import { defineConfig } from "@playwright/test";

// Browser tests against `eicvibe-web --demo`, using the installed Chrome
// (PW_CHANNEL=chromium uses Playwright's own browser after `npx playwright install`).
const port = 8799;
/** The first administrator's password, as e2e/admin.spec.ts logs in with it. */
export const ADMIN_PASSWORD = "e2e-admin-password";

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
    // The demo, with a virtual accelerator in a fresh state folder whose first admin's password is known.
    command: `npm run build && EICVIBE_VA_ADMIN_PASSWORD=${ADMIN_PASSWORD} uv run --project .. eicvibe-web --demo --va "$(mktemp -d)" --port ${port}`,
    url: `http://127.0.0.1:${port}/api/v1/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
