import { test, expect } from "@playwright/test";
import { existsSync, statSync } from "fs";
import { fileURLToPath } from "url";

// Prod vivo: FRONT_URL=https://vultus.esau.com.mx npm run test:e2e
// Par dorado real fuera de VC: GOLDEN_A/GOLDEN_B con JPEG LFW (ver fixtures/README).
const FRONT = process.env.FRONT_URL ?? "http://localhost:4321";
const GOLDEN_A =
  process.env.GOLDEN_A && existsSync(process.env.GOLDEN_A)
    ? process.env.GOLDEN_A
    : fileURLToPath(new URL("./fixtures/a.png", import.meta.url));
const GOLDEN_B =
  process.env.GOLDEN_B && existsSync(process.env.GOLDEN_B)
    ? process.env.GOLDEN_B
    : fileURLToPath(new URL("./fixtures/b.png", import.meta.url));

test("compare tracer bullet", async ({ page }) => {
  await page.goto(FRONT);
  await expect(page.getByRole("heading", { name: /Vultus/ })).toBeVisible();
});

test("compare upload 2 PNG returns queued job", async ({ page }) => {
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    Buffer.alloc(56, 0),
  ]);
  await page.goto(FRONT);
  await page
    .locator('input[name="image_a"]')
    .setInputFiles({ name: "a.png", mimeType: "image/png", buffer: png });
  await page
    .locator('input[name="image_b"]')
    .setInputFiles({ name: "b.png", mimeType: "image/png", buffer: png });
  await page.getByRole("button", { name: /Comparar/ }).click();
  // El job avanza queued->processing->done en ms con sidecar local; aceptar
  // cualquier estado con job_id evita flake sin perder el intent tracer.
  await expect(page.locator("#status")).toContainText(/job .*(queued|processing|done)/, {
    timeout: 15_000,
  });
});

test("golden pair reaches done, slider responds and download starts", async ({
  page,
}) => {
  test.slow();
  // En prod el pipeline warm tarda ~18s; en cold + cola hasta 70s. Timeout amplio sin colgar.
  const doneTimeout = process.env.FRONT_URL ? 120_000 : 80_000;
  await page.goto(FRONT);
  await page.locator('input[name="image_a"]').setInputFiles(GOLDEN_A);
  await page.locator('input[name="image_b"]').setInputFiles(GOLDEN_B);
  await page.getByRole("button", { name: /Comparar/ }).click();
  await expect(page.locator("#status")).toContainText(/done/, {
    timeout: doneTimeout,
  });
  for (const id of ["panel-uv-a", "panel-uv-b", "panel-heatmap"] as const) {
    const img = page.getByTestId(id);
    await expect(img).toBeVisible();
    await expect(img).toHaveAttribute("src", /^blob:/);
    await expect
      .poll(async () => img.evaluate((e) => (e as HTMLImageElement).naturalWidth))
      .toBeGreaterThan(0);
  }
  await page.locator("#heatmap-opacity").evaluate((el) => {
    const slider = el as HTMLInputElement;
    slider.value = "20";
    slider.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.locator("#heatmap-opacity-value")).toContainText("20%");
  await expect(page.getByTestId("panel-heatmap")).toHaveCSS("opacity", "0.2");
  const downloadLink = page.locator("#download-zip");
  await expect(downloadLink).toBeVisible();
  await expect(downloadLink).toHaveAttribute("href", /^blob:/);
  await expect(downloadLink).toHaveAttribute("download", /^result-.*\.zip$/);
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    downloadLink.click(),
  ]);
  expect(download.suggestedFilename()).toMatch(/^result-.*\.zip$/);
  const filePath = await download.path();
  expect(filePath).toBeTruthy();
  expect(statSync(filePath as string).size).toBeGreaterThan(0);
});
