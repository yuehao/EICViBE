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
  await expect(page.getByText(/ring: s 0\.00 – 105\.60 of 105\.60 m/)).toBeVisible();
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
