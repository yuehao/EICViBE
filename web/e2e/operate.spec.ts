import { expect, test } from "@playwright/test";

import { ADMIN_PASSWORD } from "../playwright.config";

test("a physicist proposes a closed bump in the Operate tab, applies it and verifies it against the BPMs", async ({ page, request }) => {
  test.setTimeout(120_000);
  expect((await request.post("/api/v1/va/login", { data: { name: "admin", password: ADMIN_PASSWORD } })).ok()).toBeTruthy();
  const made = await request.post("/api/v1/va/machines", {
    data: { name: "Operate ring", kind: "ring", target: "ring", engine: "xsuite", fidelity: "fast", particles: 50, rate: 20000, ramp_turns: 256 },
  });
  expect(made.ok()).toBeTruthy();
  const machine = (await made.json()) as { id: string };
  expect((await request.post(`/api/v1/va/machines/${machine.id}/start`, { data: { reason: "e2e" } })).ok()).toBeTruthy();
  const user = await request.post("/api/v1/va/users", { data: { name: "e2e-operator-physicist", role: "physicist" } });
  const { password } = (await user.json()) as { password: string };

  await page.goto("/");
  await page.getByRole("banner").getByLabel("Machine").selectOption(machine.id);
  await page.getByRole("tab", { name: "Operate" }).click();
  const operate = page.getByTestId("operate");
  await expect(operate.getByTestId("tune-monitor")).toContainText(/tune monitor 0\.\d{4}, 0\.\d{4}/, { timeout: 60_000 });

  // Proposing asks for a login, then shows the bump with its checks; nothing is written yet.
  await operate.getByLabel("Operation").selectOption("local_bump");
  await operate.getByLabel("Bump at").fill("dr-b41");
  await operate.getByLabel("Amplitude").fill("1");
  await operate.getByRole("button", { name: "Propose" }).click();
  const dialog = page.getByRole("dialog", { name: "Log in" });
  await expect(dialog).toContainText("To compute a proposal on Operate ring, log in.");
  await dialog.getByLabel("User name").fill("e2e-operator-physicist");
  await dialog.getByLabel("Password").fill(password);
  await dialog.getByRole("button", { name: "Log in" }).click();
  const card = operate.getByTestId("proposal").first();
  await expect(card).toContainText("A closed horizontal bump of +1.000 mm at dr-b41", { timeout: 30_000 });
  await expect(card).toContainText("proposed");
  await expect(card).toContainText("✓ amplitude reached");
  await expect(card).toContainText("✓ closed: no orbit change outside");
  await expect(card.getByTestId("proposal-chart")).toBeVisible(); // opened: the predicted orbit change
  // A horizontal bump in a ring whose RF holds the energy takes a fourth corrector: the path, and so the energy, kept.
  await expect(card.locator(".proposal-changes tbody tr")).toHaveCount(4);
  await expect(card).toContainText("energy unchanged");

  // A person applies it; once the settings have ramped, the BPMs show the change predicted.
  await card.getByRole("button", { name: "Apply" }).click();
  await expect(card).toContainText("applied");
  await expect(async () => {
    await card.getByRole("button", { name: "Verify" }).click();
    await expect(card.getByTestId("verification")).toContainText("✓ Verified", { timeout: 1_000 });
  }).toPass({ timeout: 60_000 });

  // Stopped and removed, so that it is no one's default machine afterwards.
  expect((await request.post(`/api/v1/va/machines/${machine.id}/stop`, { data: { reason: "e2e done" } })).ok()).toBeTruthy();
  await expect(async () => {
    const state = ((await (await request.get(`/api/v1/va/machines/${machine.id}`)).json()) as { state: string }).state;
    expect(state).toBe("stopped");
  }).toPass({ timeout: 30_000 });
  expect((await request.delete(`/api/v1/va/machines/${machine.id}`)).ok()).toBeTruthy();
});
