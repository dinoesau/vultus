import { test, expect, type Page } from "@playwright/test";

// Demo Obama E1-E4 del issue #122. Sin backend para E1-E3 (solo assets
// estáticos + DOM); E4 usa el gateway vivo igual que compare.spec.ts.
const FRONT = process.env.FRONT_URL ?? "http://localhost:4321";
const DEMO_BTN = "#btn-demo";

async function loadDemo(page: Page) {
  await page.goto(FRONT);
  await page.locator(DEMO_BTN).click();
  for (const id of ["preview-a", "preview-b"] as const) {
    await expect(page.getByTestId(id)).toBeVisible();
  }
}

test("E1 click demo carga Obama A/B listos para Comparar", async ({ page }) => {
  await loadDemo(page);
  for (const [id, name] of [
    ["preview-a", "demo-obama-a.jpg"],
    ["preview-b", "demo-obama-b.jpg"],
  ] as const) {
    const img = page.getByTestId(id);
    await expect(img).toHaveAttribute("src", /^blob:/);
    await expect
      .poll(async () => img.evaluate((e) => (e as HTMLImageElement).naturalWidth))
      .toBeGreaterThan(0);
    const fileName = await page.evaluate((slot) => {
      const input = document.querySelector<HTMLInputElement>(`input[name="image_${slot}"]`);
      return input?.files?.[0]?.name ?? null;
    }, id === "preview-a" ? "a" : "b");
    expect(fileName).toBe(name);
  }
  await expect(page.locator("#stage-text")).toContainText(/demo Obama A\/B cargada/);
  await expect(page.locator("#demo-credit")).toContainText(/Pete Souza/);
});

test("E2 demo reemplaza archivos previos del usuario", async ({ page }) => {
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    Buffer.alloc(56, 0),
  ]);
  const demoHits: string[] = [];
  await page.goto(FRONT);
  await page.route("**/demo/obama-*.jpg", async (route) => {
    demoHits.push(route.request().url());
    await route.continue();
  });
  await page
    .locator('input[name="image_a"]')
    .setInputFiles({ name: "mia.png", mimeType: "image/png", buffer: png });
  await page.locator(DEMO_BTN).click();
  // El par demo se pide exactamente una vez (sin doble fetch).
  await expect
    .poll(async () => demoHits.length, { timeout: 10_000 })
    .toBe(2);
  // El reemplazo se afirma con retries: el meta muestra el nombre demo.
  const metaA = page.locator("#drop-a [data-meta-a]");
  await expect(metaA).toContainText(/demo-obama-a\.jpg/, { timeout: 10_000 });
  await expect(page.getByTestId("preview-a")).toHaveAttribute("src", /^blob:/);
  const fileName = await page.evaluate(() => {
    const input = document.querySelector<HTMLInputElement>('input[name="image_a"]');
    return input?.files?.[0]?.name ?? null;
  });
  expect(fileName).toBe("demo-obama-a.jpg");
});

test("E3 boton muestra loading y bloquea doble click", async ({ page }) => {
  await page.goto(FRONT);
  let releaseGate: () => void = () => {};
  const gate = new Promise<void>((resolve) => {
    releaseGate = resolve;
  });
  const demoHits: string[] = [];
  await page.route("**/demo/obama-*.jpg", async (route) => {
    demoHits.push(route.request().url());
    await gate;
    await route.continue();
  });
  const btn = page.locator(DEMO_BTN);
  await btn.click();
  // Doble click durante la carga: el guard lo ignora (un solo par pedido).
  await btn.click({ force: true }).catch(() => {});
  // La compuerta sigue cerrada hasta que afirmamos el estado loading.
  await expect(btn).toBeDisabled({ timeout: 10_000 });
  await expect(btn).toContainText(/Cargando demo/);
  await expect(btn).toHaveAttribute("aria-busy", "true");
  releaseGate();
  await expect(btn).toBeEnabled({ timeout: 10_000 });
  await expect(btn).toContainText(/Cargar demo/);
  await expect(page.getByTestId("preview-a")).toBeVisible();
  expect(demoHits.filter((u) => u.endsWith("obama-a.jpg")).length).toBe(1);
  expect(demoHits.filter((u) => u.endsWith("obama-b.jpg")).length).toBe(1);
});

test("E4 tras demo, Comparar envia 2 JPEG y abre job", async ({ page }) => {
  await loadDemo(page);
  await page.getByRole("button", { name: /Comparar/ }).click();
  await expect(page.locator("#stage-text")).toContainText(
    /job .*(cola|enviando|procesando|listo)/,
    { timeout: 15_000 },
  );
});
