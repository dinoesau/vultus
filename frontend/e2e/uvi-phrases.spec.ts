import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";

// Story-157 rotación de frases Uvi: ambient en loop cada ~1.8s.
// E2E sin backend (rutas mockeadas); job en processing eterno para observar
// el loop, y failed terminal para el ocultado.
const FRONT = process.env.FRONT_URL ?? "http://localhost:4321";

async function mockLiveJob(page: Page): Promise<void> {
  await page.route("**/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ contract_version: 3 }),
    });
  });
  await page.route("**/v1/compare", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ job_id: "uvi-phrases-1" }),
    });
  });
}

async function fillAndCompare(page: Page): Promise<void> {
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    Buffer.alloc(56, 0),
  ]);
  await page
    .locator('input[name="image_a"]')
    .setInputFiles({ name: "a.png", mimeType: "image/png", buffer: png });
  await page
    .locator('input[name="image_b"]')
    .setInputFiles({ name: "b.png", mimeType: "image/png", buffer: png });
  await page.getByRole("button", { name: /Comparar/ }).click();
}

test("157 AC1 frases rotan en orden cada ~1.8s en loop", async ({ page }) => {
  await page.goto(FRONT);
  await mockLiveJob(page);
  await page.route("**/v1/jobs/uvi-phrases-1", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "processing" }),
    });
  });
  await fillAndCompare(page);
  const text = page.locator("#uvi-text");
  await expect(page.getByTestId("uvi")).toBeVisible({ timeout: 10_000 });
  await expect(text).toHaveText("Exploring...", { timeout: 10_000 });
  await expect(text).toHaveText("Discovering...", { timeout: 6_000 });
  await expect(text).toHaveText("Unraveling...", { timeout: 6_000 });
});

test("157 AC3 #uvi-text es live region unica mono 13px", async ({ page }) => {
  await page.goto(FRONT);
  const text = page.locator("#uvi-text");
  await expect(text).toHaveAttribute("role", "status");
  await expect(text).toHaveAttribute("aria-live", "polite");
  const liveCount = await page.evaluate(
    () => document.querySelectorAll("#uvi [aria-live]").length,
  );
  expect(liveCount).toBe(1);
  const style = await page.evaluate(() => {
    const el = document.getElementById("uvi-text");
    if (!(el instanceof HTMLElement)) return null;
    const cs = getComputedStyle(el);
    return { size: cs.fontSize, family: cs.fontFamily };
  });
  expect(style).not.toBeNull();
  expect(style!.size).toBe("13px");
  expect(style!.family).toContain("JetBrains Mono");
});

test("157 AC2 failed terminal oculta Uvi+frases y limpia el texto", async ({
  page,
}) => {
  await page.goto(FRONT);
  await mockLiveJob(page);
  await page.route("**/v1/jobs/uvi-phrases-1", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "failed" }),
    });
  });
  await fillAndCompare(page);
  await expect(page.getByTestId("uvi")).toBeHidden({ timeout: 15_000 });
  await expect(page.locator("#uvi-text")).toHaveText("", { timeout: 5_000 });
});
