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
  // The demo's supplies have placeholder transfer functions: marked as such, and no design beyond its limits.
  await expect(page.getByTestId("settings").getByTestId("source-badge")).toHaveText("placeholder");
  await expect(page.getByTestId("settings").locator(".beyond")).toHaveCount(0);
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

test("a machine's Orbit link follows the machine into the run it starts next", async ({ page, request }) => {
  await asAdmin(request);
  const made = await request.post("/api/v1/va/machines", {
    data: { name: "Restart ring", kind: "ring", target: "ring", engine: "xsuite", fidelity: "fast", particles: 8 },
  });
  expect(made.ok()).toBeTruthy();
  const machine = (await made.json()) as { id: string };
  const started = async () => {
    expect((await request.post(`/api/v1/va/machines/${machine.id}/start`, { data: { reason: "e2e" } })).ok()).toBeTruthy();
    let run: string | null = null;
    await expect(async () => {
      const entry = (await (await request.get(`/api/v1/va/machines/${machine.id}`)).json()) as { state: string; run: string | null };
      expect(entry.state).toBe("running");
      run = entry.run;
    }).toPass({ timeout: 30_000 });
    return run!;
  };
  const first = await started();

  await adminLogin(page, "admin", ADMIN_PASSWORD);
  const link = page.locator(`tr[data-machine=${machine.id}]`).getByRole("link", { name: "Orbit" });
  const href = await link.getAttribute("href");
  expect(href).toContain(`machine=${machine.id}`);
  expect(href).not.toContain("run="); // the machine, whichever run it has
  await page.goto(href!);
  const orbit = page.getByRole("tabpanel", { name: "Orbit" });
  await expect(orbit.getByLabel("Run")).toHaveValue(first);
  await expect(orbit.getByTestId("bpm-turns")).toContainText("turns", { timeout: 20_000 });

  expect((await request.post(`/api/v1/va/machines/${machine.id}/stop`, { data: { reason: "e2e" } })).ok()).toBeTruthy();
  await expect(orbit.getByTestId("bpm-state")).toHaveText("stopped", { timeout: 20_000 });
  const second = await started();
  expect(second).not.toBe(first);
  await expect(orbit.getByLabel("Run")).toHaveValue(second, { timeout: 20_000 });
  await expect(orbit.getByTestId("bpm-state")).toHaveText("running", { timeout: 20_000 });
  await expect(orbit.getByTestId("bpm-turns")).toContainText("turns", { timeout: 20_000 });

  await request.post(`/api/v1/va/machines/${machine.id}/stop`, { data: { reason: "e2e" } });
});

test("an administrator chooses a machine's errors, which it draws when it starts", async ({ page, request }) => {
  await asAdmin(request);
  const made = await request.post("/api/v1/va/machines", {
    data: { name: "Errors ring", kind: "ring", target: "ring", engine: "xsuite", fidelity: "fast", particles: 8 },
  });
  expect(made.ok()).toBeTruthy();
  const machine = (await made.json()) as { id: string };

  await adminLogin(page, "admin", ADMIN_PASSWORD);
  const row = page.locator(`tr[data-machine=${machine.id}]`);
  await row.getByRole("button", { name: "Errors" }).click();
  const editor = page.getByTestId("errors-editor");
  await expect(editor).toBeVisible();
  await editor.getByRole("button", { name: "+ Typical alignment" }).click();
  await expect(editor.getByLabel("x offset µm, row 1")).toHaveValue("200");
  await expect(editor.getByTestId("chooses-magnet-1")).toHaveText("67 magnets"); // the ring's bends, quads and sextupoles
  // A name that matches nothing is said at once, and refused when saved.
  await editor.getByLabel("Names, row 1").fill("any");
  await expect(editor.getByTestId("chooses-magnet-1")).toHaveText("none");
  await editor.getByRole("button", { name: "Save errors" }).click();
  await expect(page.getByTestId("errors-note")).toContainText("no Bend or Quadrupole or Sextupole named any here");
  await editor.getByLabel("Names, row 1").fill("");
  await expect(editor.getByTestId("chooses-magnet-1")).toHaveText("67 magnets");
  await editor.getByLabel("pitch mrad, row 1").fill("0.1");
  await editor.getByRole("button", { name: "+ Fluctuating supply" }).click();
  await editor.getByLabel("Supply names, row 1").fill("b_*");
  await editor.getByLabel("Ripple, row 1").fill("360:5");
  await editor.getByLabel("BPM rolls mrad").fill("1");
  await editor.getByLabel("Reason for the errors").fill("e2e errors");
  await editor.getByRole("button", { name: "Save errors" }).click();
  await expect(page.getByTestId("errors-note")).toContainText("draws these errors (seed");

  const entry = (await (await request.get(`/api/v1/va/machines/${machine.id}/errors`)).json()) as {
    errors: {
      magnets: { x_offset: number; roll: number; pitch: number; select: { kinds: string[] } }[];
      supplies: { rms: number; ripple: { frequency: number; amplitude: number }[]; select: { names: string[] } }[];
      bpms: { roll_rms: number };
    };
    chosen: { magnets: number[]; supplies: number[] };
  };
  const stored = entry.errors;
  expect(entry.chosen).toEqual({ magnets: [67], supplies: [1] }); // the string of b_* bends is one supply
  expect(stored.magnets[0]!.select.kinds).toEqual(["Bend", "Quadrupole", "Sextupole"]);
  expect([stored.magnets[0]!.x_offset, stored.magnets[0]!.roll, stored.magnets[0]!.pitch]).toEqual([2e-4, 5e-4, 1e-4]);
  expect(stored.supplies[0]!.select.names).toEqual(["b_*"]);
  expect(stored.supplies[0]!.rms).toBeCloseTo(1e-5, 12);
  expect(stored.supplies[0]!.ripple).toEqual([{ frequency: 360, amplitude: 5e-6 }]);
  expect(stored.bpms.roll_rms).toBe(1e-3);

  // Running, it has drawn them: they cannot change until it stops.
  await page.getByRole("button", { name: "Machines" }).click();
  await row.getByRole("button", { name: "Start" }).click();
  await expect(row.getByTestId("machine-state")).toHaveText("running", { timeout: 30_000 });
  await row.getByRole("button", { name: "Errors" }).click();
  await expect(editor).toContainText("Stop Errors ring to change its errors");
  await expect(editor.getByRole("button", { name: "Save errors" })).toBeDisabled();
  await request.post(`/api/v1/va/machines/${machine.id}/stop`, { data: { reason: "e2e" } });
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

test("an administrator chooses the AI's model: local by default, or a provider with a key kept unseen", async ({ browser, request }) => {
  await asAdmin(request);
  const password = await addUser(request, "ai-operator", "operator");
  const operator = await browser.newPage();
  await adminLogin(operator, "ai-operator", password);
  await expect(operator.getByRole("button", { name: "Machines" })).toBeVisible();
  await expect(operator.getByRole("button", { name: "AI", exact: true })).toHaveCount(0); // administrators only
  await operator.close();

  const page = await browser.newPage();
  await adminLogin(page, "admin", ADMIN_PASSWORD);
  await page.getByRole("button", { name: "AI", exact: true }).click();
  const section = page.getByTestId("ai-settings");
  await expect(page.getByTestId("ai-now")).toContainText("no model server at http://127.0.0.1:9");
  await expect(section.getByRole("radio", { name: "Local model (Ollama), the default" })).toBeChecked();
  await expect(page.getByTestId("ai-note")).toContainText("nothing leaves the host");

  await section.getByRole("radio", { name: "Anthropic Claude" }).check();
  await expect(page.getByTestId("ai-note")).toContainText("are sent to Anthropic");
  await expect(page.getByTestId("ai-note")).toContainText("A ChatGPT or Claude subscription does not let other applications use its models");
  await section.getByLabel("URL").fill("http://127.0.0.1:9"); // nothing there: the test stays offline
  await section.getByLabel("API key").fill("sk-ant-e2e-secret-9876");
  await section.getByRole("button", { name: "Test" }).click();
  await expect(page.getByTestId("ai-test")).toContainText("Does not answer: Anthropic cannot be reached");
  await section.getByLabel("Reason").fill("e2e: try Claude");
  await section.getByRole("button", { name: "Save" }).click();
  await expect(page.getByTestId("ai-now")).toContainText("claude-opus-5-5 on 127.0.0.1");
  await expect(section).toContainText("Key: stored (…9876)");
  const settings = await request.get("/api/v1/va/ai");
  expect(await settings.text()).not.toContain("sk-ant-e2e-secret-9876");

  await page.getByRole("button", { name: "Log", exact: true }).click();
  await page.getByLabel("Log", { exact: true }).selectOption("actions");
  const actions = page.getByTestId("actions");
  await expect(actions).toContainText("ai settings");
  await expect(actions).toContainText("e2e: try Claude");
  await expect(actions).not.toContainText("sk-ant-e2e-secret-9876");

  // Back to the default, for the other tests.
  await page.getByRole("button", { name: "AI", exact: true }).click();
  await section.getByRole("radio", { name: "Local model (Ollama), the default" }).check();
  await section.getByLabel("URL").fill("http://127.0.0.1:9");
  await section.getByRole("button", { name: "Save" }).click();
  await expect(page.getByTestId("ai-now")).toContainText("no model server at http://127.0.0.1:9");
});

test("a magnet on a string shows its supply's setting, which only the supply's own window changes", async ({ page, request }) => {
  await asAdmin(request);
  const made = await request.post("/api/v1/va/machines", {
    data: { name: "String ring", kind: "ring", target: "ring", engine: "xsuite", fidelity: "fast", particles: 8 },
  });
  expect(made.ok()).toBeTruthy();
  const machine = (await made.json()) as { id: string };
  const password = await addUser(request, "e2e-string-physicist", "physicist");

  await page.goto("/#branch=ring");
  await page.getByRole("banner").getByLabel("Machine").selectOption(machine.id);
  await page.getByLabel("Find element").fill("QF_1");
  await page.getByRole("option", { name: /\bQF_1\b/ }).first().click();
  const setpoints = page.getByTestId("setpoints");
  await expect(setpoints).toContainText("dr-q4-ps:sp");
  // The string's setting drives eleven quadrupoles: shown here, not set from one of them.
  await expect(setpoints.getByTestId("setpoint-shared")).toContainText("drives 11 magnets in series");
  await expect(setpoints.getByRole("button", { name: "Set dr-q4-ps:sp" })).toHaveCount(0);
  await expect(setpoints.getByLabel("New value of dr-q4-ps:sp")).toHaveCount(0);

  await setpoints.getByRole("button", { name: "Power supply dr-q4-ps" }).click();
  const supply = page.getByRole("dialog", { name: "Power supply dr-q4-ps" });
  await expect(supply).toContainText("It drives 11 magnets in series; a change moves them all");
  await expect(supply.getByTestId("supply-magnets")).toContainText("QF_1");
  await supply.getByLabel("New value of dr-q4-ps:sp").fill("27");
  await supply.getByRole("button", { name: "Set all 11" }).click();
  const login = page.getByRole("dialog", { name: "Log in" });
  await login.getByLabel("User name").fill("e2e-string-physicist");
  await login.getByLabel("Password").fill(password);
  await login.getByRole("button", { name: "Log in" }).click();
  await expect(supply.getByTestId("setpoint-outcome")).toContainText("Set, logged as change");
  await supply.getByRole("button", { name: "Close" }).click();

  // Every magnet of the string has the new value.
  await page.getByLabel("Find element").fill("QF_2");
  await page.getByRole("option", { name: /\bQF_2\b/ }).first().click();
  await expect(setpoints).toContainText("27 A");
  await expect(setpoints).toContainText("set by e2e-string-physicist");
});
