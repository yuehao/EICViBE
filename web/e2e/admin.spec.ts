import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

import { ADMIN_PASSWORD } from "../playwright.config";

/** Log ``request`` in as the administrator, for setting things up through the API. */
async function asAdmin(request: APIRequestContext) {
  const response = await request.post("/api/v1/va/login", { data: { name: "admin", password: ADMIN_PASSWORD } });
  expect(response.ok()).toBeTruthy();
}

async function addUser(request: APIRequestContext, name: string, role: string): Promise<string> {
  const response = await request.post("/api/v1/va/users", { data: { name, role } });
  expect(response.ok()).toBeTruthy();
  return ((await response.json()) as { password: string }).password;
}

async function adminLogin(page: Page, name: string, password: string) {
  await page.goto("/admin.html");
  const gate = page.getByTestId("admin-gate");
  await gate.getByLabel("User name").fill(name);
  await gate.getByLabel("Password").fill(password);
  await gate.getByRole("button", { name: "Log in" }).click();
}

test("the admin page shows only a login, and only operators and administrators get in", async ({ page, request }) => {
  await page.goto("/");
  await expect(page.locator("[data-testid=element-table] .tr").first()).toBeVisible();
  await expect(page.locator("a[href*=admin]")).toHaveCount(0); // nothing in the viewer leads there

  await page.goto("/admin.html");
  await expect(page.getByTestId("admin-gate")).toContainText("EICViBE administration");
  await expect(page.getByTestId("machines")).toHaveCount(0);

  await asAdmin(request);
  const password = await addUser(request, "gate-physicist", "physicist");
  await adminLogin(page, "gate-physicist", password);
  await expect(page.getByTestId("admin-gate")).toContainText("This page is for operators and administrators");
  await expect(page.getByTestId("machines")).toHaveCount(0);
});

test("an administrator starts a machine, changes a setting, and the log keeps it", async ({ page }) => {
  await adminLogin(page, "admin", ADMIN_PASSWORD);
  const form = page.getByRole("form", { name: "New machine" });
  await form.getByLabel("Name").fill("E2E ring");
  await expect(form.getByRole("button", { name: "Create" })).toBeDisabled(); // what it tracks is chosen, not assumed
  await form.getByLabel("Ring or route").selectOption("ring");
  await form.getByLabel("Engine").selectOption("xsuite");
  await form.getByLabel("Fidelity").selectOption("fast");
  await form.getByRole("button", { name: "Create" }).click();
  const row = page.locator("tr[data-machine=e2e-ring]");
  await expect(row).toContainText("shared");
  await expect(row).toContainText("changes ramp over 1024 turns"); // the default: a change never jumps
  await row.getByRole("button", { name: "Start" }).click();
  await expect(row.getByTestId("machine-state")).toHaveText("running", { timeout: 30_000 });

  await row.getByRole("button", { name: "Settings" }).click();
  await page.getByLabel("Filter settings").fill("dr-t8-ps:sp:h");
  await page.getByLabel("New value of dr-t8-ps:sp:h").fill("1.5");
  await page.getByLabel("Reason").fill("e2e bump");
  await page.getByRole("button", { name: "Apply as one change" }).click();
  await expect(page.getByTestId("write-outcome")).toContainText("Applied");
  await expect(page.getByTestId("settings").locator("tr.changed")).toContainText("admin");
  await page.getByLabel("New value of dr-t8-ps:sp:h").fill("500");
  await page.getByRole("button", { name: "Apply as one change" }).click();
  await expect(page.getByTestId("write-outcome")).toContainText("outside the limits");

  await page.getByRole("button", { name: "Log", exact: true }).click();
  const log = page.getByTestId("log");
  await expect(log.locator("tbody tr").first()).toContainText("outside the limits");
  await expect(log.locator("tbody tr").nth(1)).toContainText("e2e bump");

  await page.getByRole("button", { name: "Users" }).click();
  await page.getByLabel("New user name").fill("e2e-operator");
  await page.getByLabel("New user's role").selectOption("operator");
  await page.getByRole("button", { name: "Add user" }).click();
  await expect(page.getByTestId("new-password")).toContainText("e2e-operator");

  await page.getByRole("button", { name: "Machines" }).click();
  await page.locator("tr[data-machine=e2e-ring]").getByRole("button", { name: "Stop" }).click();
  await expect(page.locator("tr[data-machine=e2e-ring]").getByTestId("machine-state")).toHaveText("stopped", {
    timeout: 30_000,
  });
});

test("a user views without logging in, and logs in only to change a setting", async ({ page, request }) => {
  await asAdmin(request);
  const made = await request.post("/api/v1/va/machines", {
    data: { name: "Viewer ring", kind: "ring", target: "ring", engine: "xsuite", fidelity: "fast", particles: 8 },
  });
  expect(made.ok()).toBeTruthy();
  const password = await addUser(request, "e2e-physicist", "physicist");

  await page.goto("/#branch=injection");
  await page.getByRole("banner").getByLabel("Machine").selectOption("viewer-ring");
  const ring = page.getByRole("navigation", { name: "Branches" }).getByRole("button", { name: /^ring/ });
  await expect(ring).toHaveAttribute("aria-pressed", "true"); // choosing a machine shows what it tracks
  await page.getByLabel("Find element").fill("COR_1");
  await page.getByRole("option", { name: /\bCOR_1\b/ }).first().click();
  const setpoints = page.getByTestId("setpoints");
  await expect(setpoints).toContainText("Setpoints on Viewer ring");
  await expect(setpoints).toContainText("dr-t8-ps:sp:h");
  await expect(page.getByRole("dialog", { name: "Log in" })).toHaveCount(0); // viewing asks for nothing

  await setpoints.getByLabel("New value of dr-t8-ps:sp:h").fill("1.5");
  await setpoints.getByRole("button", { name: "Set dr-t8-ps:sp:h" }).click();
  const dialog = page.getByRole("dialog", { name: "Log in" });
  await expect(dialog).toContainText("To set dr-t8-ps:sp:h on Viewer ring, log in.");
  await dialog.getByLabel("User name").fill("e2e-physicist");
  await dialog.getByLabel("Password").fill(password);
  await dialog.getByRole("button", { name: "Log in" }).click();
  await expect(page.getByTestId("setpoint-outcome")).toContainText("Set, logged as change");
  await expect(page.getByTestId("setpoint-outcome")).toContainText("It ramps in over 1024 turns.");
  await expect(setpoints).toContainText("1.5 A");
  await expect(setpoints).toContainText("set by e2e-physicist");
  await expect(page.getByRole("banner").getByRole("button", { name: "e2e-physicist" })).toBeVisible();

  // Logged in now: a second change asks nothing; a sandbox of one's own starts from the machine's settings.
  await setpoints.getByLabel("New value of dr-t8-ps:sp:v").fill("-0.5");
  await setpoints.getByRole("button", { name: "Set dr-t8-ps:sp:v" }).click();
  await expect(setpoints).toContainText("-0.5 A");
  await page.getByRole("banner").getByRole("button", { name: "Sandbox" }).click();
  const chooser = page.getByRole("banner").getByLabel("Machine");
  await expect(chooser.locator("option:checked")).toContainText("e2e-physicist's sandbox");
  await expect(setpoints).toContainText("1.5 A"); // the sandbox's copy of the settings
});
