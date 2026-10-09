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
  await expect(page.locator("#stage-text")).toContainText(/job .*(queued|processing|done|cola|procesando|listo)/, {
    timeout: 15_000,
  });
});

test("golden pair reaches done, 6 panels plus 2 viewers, zero heatmap, download starts", async ({
  page,
}) => {
  test.slow();
  // En prod el pipeline warm tarda ~18s; en cold + cola hasta 70s. Timeout amplio sin colgar.
  const doneTimeout = process.env.FRONT_URL ? 120_000 : 80_000;
  await page.goto(FRONT);
  await page.locator('input[name="image_a"]').setInputFiles(GOLDEN_A);
  await page.locator('input[name="image_b"]').setInputFiles(GOLDEN_B);
  await page.getByRole("button", { name: /Comparar/ }).click();
  await expect(page.locator("#stage-text")).toContainText(/done|listo/, {
    timeout: doneTimeout,
  });
  // Zip-8 sin heatmap (ADR-008 + ADR-012): 6 paneles uv/pbr/render con blob src.
  for (const id of ["panel-uv-a", "panel-uv-b", "panel-pbr-a", "panel-pbr-b", "panel-render-a", "panel-render-b"] as const) {
    const img = page.getByTestId(id);
    await expect(img).toBeVisible();
    await expect(img).toHaveAttribute("src", /^blob:/);
    await expect
      .poll(async () => img.evaluate((e) => (e as HTMLImageElement).naturalWidth))
      .toBeGreaterThan(0);
  }
  // Cero heatmap: ni panel ni slider sobreviven al corte (Wave 5).
  await expect(page.getByTestId("panel-heatmap")).toHaveCount(0);
  await expect(page.locator("#heatmap-opacity")).toHaveCount(0);
  await expect(page.locator("#heatmap-opacity-value")).toHaveCount(0);
  for (const id of ["viewer-3d-a", "viewer-3d-b"] as const) {
    const viewer = page.getByTestId(id);
    await expect(viewer).toBeVisible();
  }
  await expect(page.locator("#viewer-3d-status")).toContainText(/cara real lista/);
  // El loader debe parsear los GLB reales con PBR: los canvas dejan de ser
  // fondo. Gate estricto (Wave 6-fix): cada captura supera el umbral minimo
  // de bytes (descarta canvas vacio o captura fallida) y las dos caras
  // difieren entre si (varianza: dos triangulos sinteticos identicos darian
  // bytes iguales y fallan; la evidencia de cara real vive en
  // scripts/e2e-flame-real.py CHECK 7-ssim).
  const MIN_CANVAS_PNG_BYTES = 1500;
  const shotA = await page.getByTestId("viewer-3d-a").screenshot();
  const shotB = await page.getByTestId("viewer-3d-b").screenshot();
  expect(shotA.length).toBeGreaterThan(MIN_CANVAS_PNG_BYTES);
  expect(shotB.length).toBeGreaterThan(MIN_CANVAS_PNG_BYTES);
  expect(shotA.equals(shotB)).toBe(false);
  for (const id of ["download-mesh-a", "download-mesh-b"] as const) {
    const link = page.locator(`#${id}`);
    await expect(link).toBeVisible();
    await expect(link).toHaveAttribute("href", /^blob:/);
  }
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
