import { test, expect } from "@playwright/test";

test("compare tracer bullet", async ({ page }) => {
  await page.goto("http://localhost:4321");
  await expect(page.getByRole("heading", { name: /Vultus/ })).toBeVisible();
});

test("compare upload 2 PNG returns queued job", async ({ page }) => {
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    Buffer.alloc(56, 0),
  ]);
  await page.goto("http://localhost:4321");
  await page
    .locator('input[name="image_a"]')
    .setInputFiles({ name: "a.png", mimeType: "image/png", buffer: png });
  await page
    .locator('input[name="image_b"]')
    .setInputFiles({ name: "b.png", mimeType: "image/png", buffer: png });
  await page.getByRole("button", { name: /Comparar/ }).click();
  await expect(page.locator("#status")).toContainText(/job .*queued/, {
    timeout: 15_000,
  });
});
