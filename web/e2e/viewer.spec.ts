import { expect, test, type Page } from "@playwright/test";

const rows = (page: Page) => page.locator("[data-testid=element-table] .tr:not(.head)");
const branchButton = (page: Page, name: string) =>
  page.getByRole("navigation", { name: "Branches" }).getByRole("button", { name: new RegExp(`^${name}`) });

async function open(page: Page, hash = "") {
  await page.goto(`/${hash}`);
  await expect(rows(page).first()).toBeVisible();
}

/** Select an element by its lattice name through the search box. */
async function find(page: Page, name: string) {
  await page.getByLabel("Find element").fill(name);
  await page.getByRole("option", { name: new RegExp(`\\b${name}\\b`) }).first().click();
  await expect(page.getByTestId("inspector").getByText(name, { exact: true }).first()).toBeVisible();
}

test("shows the floor plan, the linear view and the element table", async ({ page }) => {
  await open(page);
  await expect(branchButton(page, "ring")).toHaveAttribute("aria-pressed", "true");
  for (const id of ["floor-canvas", "linear-canvas"]) {
    const drawn = await page.getByTestId(id).evaluate((canvas: HTMLCanvasElement) => {
      const { data } = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height);
      const colours = new Set<number>();
      for (let i = 0; i < data.length; i += 4 * 97) colours.add((data[i]! << 16) | (data[i + 1]! << 8) | data[i + 2]!);
      return colours.size;
    });
    expect(drawn, id).toBeGreaterThan(4);
  }
  await expect(page.locator(".range-readout")).toHaveText("s 0.00 – 105.60 of 105.60 m");
  await expect(rows(page).filter({ hasText: "Drift" })).toHaveCount(0); // drifts are not listed
  await expect(page.locator(".chip", { hasText: "Drift" })).toHaveCount(0);
});

test("a selection is linked across views and survives a reload of its URL", async ({ page }) => {
  await open(page);
  await rows(page).filter({ hasText: "QD_3" }).click();
  const inspector = page.getByTestId("inspector");
  await expect(inspector.getByRole("heading", { name: /^dr-q\d+/ })).toBeVisible(); // its EIC name
  await expect(inspector.getByText("QD_3", { exact: true })).toBeVisible(); // and its lattice name
  await expect(inspector.getByText("dr-")).not.toHaveCount(0); // device names from the registry
  await expect(inspector.getByText("string QD")).toBeVisible(); // a quadrupole family on one supply
  await expect(inspector.getByText(/magnets in series/)).toBeVisible();
  await expect(inspector.getByText(/field = 0\.1 T\/m\/A × I/)).toBeVisible(); // its transfer function
  await expect(page).toHaveURL(/sel=occ_/);
  const url = page.url();

  const other = await page.context().newPage();
  await other.goto(url);
  await expect(other.getByTestId("inspector").getByText("QD_3", { exact: true })).toBeVisible();
  await expect(other.locator(".tr.selected")).toContainText("QD_3");
});

test("two users browse independently", async ({ browser }) => {
  const alice = await (await browser.newContext()).newPage();
  const bob = await (await browser.newContext()).newPage();
  await open(alice);
  await open(bob);
  await branchButton(alice, "injection").click();
  await find(alice, "BI_1"); // only in the injection line
  await expect(branchButton(alice, "injection")).toHaveAttribute("aria-pressed", "true");
  await bob.waitForTimeout(300);
  await expect(branchButton(bob, "ring")).toHaveAttribute("aria-pressed", "true");
  await expect(bob.getByText("Select an element in any view to inspect it.")).toBeVisible();
  await bob.getByLabel("Filter elements").fill("SF");
  await expect(alice.getByLabel("Filter elements")).toHaveValue("");
});

test("a fork leads to the branch it feeds", async ({ page }) => {
  await open(page);
  await find(page, "fork_ring_to_extraction");
  await page.getByTestId("inspector").getByRole("button", { name: "extraction" }).first().click();
  await expect(branchButton(page, "extraction")).toHaveAttribute("aria-pressed", "true");
  await expect(rows(page).first()).toContainText("X_START");
});

test("hovering the floor plan names the element under the cursor", async ({ page }) => {
  await open(page);
  await find(page, "B_5"); // zooms the floor plan onto the dipole
  const canvas = page.getByTestId("floor-canvas");
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await expect(page.locator(".floor .tooltip")).toContainText("B_5");
  await expect(page.locator(".floor .tooltip")).toContainText("angle = 0.2618 rad");
});

test("the optics panel follows the branch, with a ring's tunes and chromaticities", async ({ page }) => {
  await open(page);
  const summary = page.getByTestId("optics-summary");
  await expect(summary).toHaveText(/^Q \d+\.\d+, \d+\.\d+ · ξ -?\d/, { timeout: 20_000 });
  const colours = await page.getByTestId("optics-canvas").evaluate((canvas: HTMLCanvasElement) => {
    const { data } = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height);
    const found = new Set<number>();
    for (let i = 0; i < data.length; i += 4 * 13) found.add((data[i]! << 16) | (data[i + 1]! << 8) | data[i + 2]!);
    return found.size;
  });
  expect(colours).toBeGreaterThan(4);
  await expect(page.getByLabel("Optics engine").locator("option")).toContainText(["Xsuite", "TrackPad"]);

  await page.getByLabel("Optics fidelity").selectOption("precise");
  await expect(page).toHaveURL(/fidelity=precise/);
  await expect(summary).toHaveText(/^Q /, { timeout: 20_000 });

  await branchButton(page, "extraction").click();
  await expect(summary).toHaveText(/^from fork .* in ring · μ/, { timeout: 20_000 });
  await branchButton(page, "injection").click();
  await expect(summary).toHaveText(/^from initial conditions/, { timeout: 20_000 });
});

test("the optics may have an operating mode's settings instead of the lattice's", async ({ page }) => {
  // The demo is a lattice file, which has no modes; a machine folder's settings/<mode>.yaml is stood in
  // for here, checking what the page asks for and shows (the server's side has its own tests).
  await page.route(
    (url) => /\/api\/v1\/lattices\/[^/]+$/.test(url.pathname),
    async (route) => {
      const summary = await (await route.fetch()).json();
      await route.fulfill({ json: { ...summary, modes: ["matched"] } });
    },
  );
  await page.route(/\/optics\?.*mode=matched/, async (route) => {
    const url = new URL(route.request().url());
    url.searchParams.delete("mode");
    const optics = await (await route.fetch({ url: url.toString() })).json();
    await route.fulfill({ json: { ...optics, mode: "matched", tune_x: 0.4321 } });
  });
  await open(page);
  const summary = page.getByTestId("optics-summary");
  await expect(summary).toHaveText(/^Q \d+\.\d+/, { timeout: 20_000 });
  const settings = page.getByLabel("Optics settings");
  await expect(settings.locator("option")).toHaveText(["design settings", "mode matched"]);
  await settings.selectOption("matched");
  await expect(page).toHaveURL(/mode=matched/);
  await expect(summary).toHaveText(/^Q 0\.4321,/);
  await settings.selectOption("");
  await expect(summary).not.toHaveText(/^Q 0\.4321,/);
});

test("hovering the optics names the element and reads the optics there", async ({ page }) => {
  await open(page);
  await expect(page.getByTestId("optics-summary")).toHaveText(/^Q /, { timeout: 20_000 });
  const canvas = page.getByTestId("optics-canvas");
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
  const tip = page.locator(".optics-pane .optics .optics-plot .tooltip");
  await expect(tip).toBeVisible();
  await expect(tip).toContainText(/βx [\d.]+ m {3}βy [\d.]+ m/);
  await expect(tip).toContainText("μx");
});

test("dragging zooms to an s range in the lattice strip and the optics alike; right click undoes", async ({ page }) => {
  await open(page);
  await expect(page.getByTestId("optics-summary")).toHaveText(/^Q /, { timeout: 20_000 });
  const readout = page.locator(".range-readout");
  await expect(readout).toHaveText("s 0.00 – 105.60 of 105.60 m");
  const drag = async (testId: string, from: number, to: number) => {
    const box = (await page.getByTestId(testId).boundingBox())!;
    const x = (f: number) => box.x + 60 + (box.width - 120) * f; // inside the shared 60 px margins
    await page.mouse.move(x(from), box.y + box.height * 0.6);
    await page.mouse.down();
    await page.mouse.move(x((from + to) / 2), box.y + box.height * 0.6, { steps: 3 });
    await page.mouse.move(x(to), box.y + box.height * 0.6, { steps: 3 });
    await page.mouse.up();
  };
  await drag("optics-canvas", 0.25, 0.5);
  await expect(readout).toHaveText(/^s 26\.\d\d – 52\.\d\d of 105\.60 m$/);
  await drag("linear-canvas", 0.5, 1.0);
  await expect(readout).toHaveText(/^s 39\.\d\d – 52\.\d\d of 105\.60 m$/);
  const plot = (await page.getByTestId("optics-canvas").boundingBox())!;
  await page.mouse.click(plot.x + plot.width / 2, plot.y + plot.height / 2, { button: "right" });
  await expect(readout).toHaveText(/^s 26\.\d\d – 52\.\d\d of 105\.60 m$/);
  await page.mouse.click(plot.x + plot.width / 2, plot.y + plot.height / 2, { button: "right" });
  await expect(readout).toHaveText("s 0.00 – 105.60 of 105.60 m");
});

test("the Orbit tab follows the demo ring's run, orbits every N turns, and pauses on its history", async ({ page }) => {
  await open(page);
  await page.getByRole("tab", { name: "Orbit" }).click();
  await expect(page).toHaveURL(/tab=orbit/);
  await expect(page.getByTestId("optics-canvas")).toBeHidden(); // the tabs share the panel
  const turns = page.getByTestId("bpm-turns");
  await expect(page.getByTestId("bpm-state")).toHaveText("running", { timeout: 30_000 });
  await expect(turns).toHaveText(/^turns \d+–\d+ of \d+/, { timeout: 30_000 });
  const span = async () => {
    const found = /^turns (\d+)–(\d+) of (\d+)/.exec((await turns.textContent()) ?? "");
    return found ? found.slice(1).map(Number) : [0, 0, 0];
  };
  const [oldest, newest] = await span();
  expect(newest! - oldest!).toBe(20); // the last 5 orbits, 5 turns apart
  await expect.poll(async () => (await span())[1], { timeout: 10_000 }).toBeGreaterThan(newest!); // live
  const colours = await page.getByTestId("bpm-canvas").evaluate((canvas: HTMLCanvasElement) => {
    const { data } = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height);
    const found = new Set<number>();
    for (let i = 0; i < data.length; i += 4 * 13) found.add((data[i]! << 16) | (data[i + 1]! << 8) | data[i + 2]!);
    return found.size;
  });
  expect(colours).toBeGreaterThan(8); // faded copies of two colours, and the axes

  await page.getByLabel("Turns between orbits").fill("10");
  await expect.poll(async () => { const [a, b] = await span(); return b! - a!; }).toBe(40);

  await page.getByRole("button", { name: "Pause" }).click();
  const paused = (await span())[1]!;
  await page.waitForTimeout(1500);
  expect((await span())[1]).toBe(paused);
  await page.getByRole("button", { name: "Back" }).click();
  await expect.poll(async () => (await span())[1]).toBe(paused - 10);
  await page.getByRole("button", { name: "Live" }).click();
  await expect.poll(async () => (await span())[1], { timeout: 10_000 }).toBeGreaterThan(paused);

  // As a BPM system keeps them: the buffer's latest turns, the slow orbit, and captures frozen by kicks.
  const data = page.getByLabel("BPM data");
  const kick = data.locator("option", { hasText: /^kick at turn/ }).first();
  await expect(kick).toBeAttached({ timeout: 20_000 });
  await data.selectOption((await kick.getAttribute("value"))!);
  await expect(page.getByTestId("bpm-state")).toHaveText("capture");
  await expect(turns).toHaveText(/^turns \d+–\d+ of \d+–\d+/);
  await data.selectOption("slow");
  await expect(turns).toHaveText(/^samples \d+–\d+ of \d+ \(1 s apart\)/, { timeout: 10_000 });
});

test("the Check tab measures the demo ring's tunes after a kick and finds the model's", async ({ page }) => {
  test.setTimeout(90_000); // it waits for a kick of the demo ring to be captured
  await open(page);
  await page.getByRole("tab", { name: "Check" }).click();
  await expect(page).toHaveURL(/tab=check/);
  // The demo ring is kicked by 0.15 mrad every 2048 turns; its kicks' captures are what the check reads.
  const x = page.getByTestId("check-x");
  await expect(x).toBeVisible({ timeout: 60_000 });
  const summary = page.getByTestId("check-summary");
  await expect(async () => {
    if (!((await summary.textContent()) ?? "").includes("of the capture of turn")) {
      await page.getByRole("button", { name: "Check again" }).click();
      throw new Error("no kick captured yet");
    }
  }).toPass({ timeout: 60_000, intervals: [2000] });
  const cells = async () => (await x.locator("td").allTextContents()).map((t) => t.trim());
  const [measured, model, , difference] = await cells();
  expect(measured).toMatch(/^0\.\d{4} \(\d+\.\d{4}\)$/);
  expect(Number(model)).toBeGreaterThan(0);
  expect(Math.abs(Number(difference))).toBeLessThan(2e-3); // the tracking gives the model's tune
  await expect(x.locator("td").nth(3)).toHaveClass(/check-good/);
  // The AI features are on by default; with no model answering, they say so and do nothing.
  await expect(page.getByRole("button", { name: "✦ Explain" })).toBeDisabled();
  await page.getByRole("button", { name: "✦ Ask" }).click();
  await expect(page.getByRole("dialog", { name: "Ask the model" })).toContainText("No model answers now");
  await page.getByRole("button", { name: "✦ Ask" }).click();
  for (const chart of ["check-spectrum", "check-drift", "check-phase", "check-beta", "check-orbit"])
    await expect(page.getByTestId(chart)).toBeVisible();

  // Tune against amplitude: single particles tracked by the model's engine.
  await page.getByLabel("tune vs amplitude").check();
  await expect(page.getByTestId("check-detuning")).toBeVisible({ timeout: 60_000 });
  await page.getByLabel("Data checked").selectOption("buffer");
  await expect(page.getByTestId("check-summary")).not.toContainText("of the capture of turn", { timeout: 60_000 });

  // The same on a page of its own, with room for the charts.
  const link = page.getByRole("link", { name: "Full page ↗" });
  const href = await link.getAttribute("href");
  expect(href).toMatch(/^check\.html#run=ring-/);
  await page.goto(`/${href}`);
  await expect(page.getByTestId("check-x")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByLabel("Data checked")).toHaveValue("kick");
});

test("with a local model the viewer offers Ask and Explain, labelled as the model's", async ({ page }) => {
  // The model server is stood in for: this checks what the page sends and shows, not the model.
  const asked: Record<string, unknown>[] = [];
  await page.route("**/api/v1/ai", (route) =>
    route.fulfill({
      json: { enabled: true, available: true, provider: "local", model: "fake:27b", where: "on this server", reason: "fake:27b at a test server" },
    }),
  );
  await page.route("**/api/v1/ai/ask", (route) => {
    asked.push(route.request().postDataJSON() as Record<string, unknown>);
    return route.fulfill({ contentType: "text/plain", body: "QF_1 is a focusing quadrupole of the ring." });
  });
  await page.route("**/api/v1/ai/explain-check", (route) => {
    asked.push(route.request().postDataJSON() as Record<string, unknown>);
    return route.fulfill({ contentType: "text/plain", body: "The tracking agrees with the model." });
  });
  await open(page);
  await find(page, "QF_1");
  await page.getByRole("button", { name: "✦ Ask" }).click();
  await page.getByLabel("Question").fill("What is this element?");
  await page.getByRole("dialog", { name: "Ask the model" }).getByRole("button", { name: "Ask", exact: true }).click();
  const answer = page.getByTestId("ai-ask-answer");
  await expect(answer).toContainText("QF_1 is a focusing quadrupole of the ring.");
  await expect(answer).toContainText("fake:27b, on this server: may be wrong");
  expect(asked[0]).toMatchObject({ question: "What is this element?", tab: "optics" });
  expect(asked[0]!.occurrence).toBeTruthy();
  expect(String(asked[0]!.branch)).toMatch(/^br_/);

  await page.getByRole("button", { name: "✦ Ask" }).click(); // closes it
  await page.getByRole("tab", { name: "Check" }).click();
  await expect(page.getByTestId("check-x")).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "✦ Explain" }).click();
  await expect(page.getByTestId("check-explanation")).toContainText("The tracking agrees with the model.");
  expect(asked[1]).toMatchObject({ source: "kick", turns: 1024, detuning: false });
});

test("a route joins its branches: one strip, its optics, and its shots' trajectories", async ({ page }) => {
  await open(page);
  const route = page.getByRole("navigation", { name: "Routes" }).getByRole("button", { name: /transfer/ });
  await route.click();
  await expect(route).toHaveAttribute("aria-pressed", "true");
  await expect(page).toHaveURL(/branch=route(:|%3A)transfer/);
  await expect(page.locator(".range-readout")).toHaveText("s 0.00 – 90.40 of 90.40 m");
  await expect(page.getByTestId("optics-summary")).toHaveText(/^from injection's initial conditions · μ/, {
    timeout: 20_000,
  });
  await find(page, "BPM_3"); // a ring BPM the route passes: selected on the route, which stays shown
  await expect(route).toHaveAttribute("aria-pressed", "true");
  await expect(page).toHaveURL(/sel=occ_/);
  await page.reload();
  await expect(route).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("inspector").getByText("BPM_3", { exact: true }).first()).toBeVisible();

  await page.getByRole("tab", { name: "Orbit" }).click();
  await expect(page.getByTestId("bpm-state")).toHaveText("running", { timeout: 30_000 });
  await expect(page.getByTestId("bpm-turns")).toHaveText(/^shots \d+–\d+ of \d+/, { timeout: 30_000 });
  await expect(page.getByLabel("Turns between orbits")).toBeVisible();
});
