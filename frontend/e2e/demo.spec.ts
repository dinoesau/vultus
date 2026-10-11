import { test, expect, type Page } from "@playwright/test";
import { DEMO_LOAD_ERROR_MESSAGE } from "../src/components/demo-loader.ts";

// Demo Obama E1-E4. Sin backend para E1-E3 (solo assets
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

test("154 header pill es boton demo canonico naranja sin TTL", async ({ page }) => {
  await page.goto(FRONT);
  const headerBtn = page.locator('#intake .hero-card-head #btn-demo');
  await expect(headerBtn).toBeVisible();
  await expect(headerBtn).toHaveAttribute("data-testid", "btn-demo");
  await expect(headerBtn).toContainText(/Cargar demo/);
  await expect(headerBtn).not.toContainText(/TTL/);
  // Sin IDs duplicados: un solo #btn-demo canonico en todo el DOM.
  await expect(page.locator("#btn-demo")).toHaveCount(1);
  // El blanco viejo sigue vivo para story-155 con id temporal.
  await expect(page.locator("#btn-demo-legacy")).toBeVisible();
});

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

test("E3 boton muestra loading y el guard ignora clicks concurrentes", async ({ page }) => {
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
  // Dispatch sintético (no click real): atraviesa el `disabled` del botón,
  // así lo que se prueba es el guard `demoLoading` del código y no la
  // supresión de clicks que ya hace el navegador. Sin guard pediría 6.
  const dispatchClick = () =>
    page.evaluate(() => {
      document
        .getElementById("btn-demo")
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
  await dispatchClick();
  await dispatchClick();
  await dispatchClick();
  await expect.poll(async () => demoHits.length, { timeout: 10_000 }).toBe(2);
  const btn = page.locator(DEMO_BTN);
  await expect(btn).toBeDisabled();
  await expect(btn).toContainText(/Cargando demo/);
  await expect(btn).toHaveAttribute("aria-busy", "true");
  releaseGate();
  await expect(btn).toBeEnabled({ timeout: 10_000 });
  await expect(btn).toContainText(/Cargar demo/);
  await expect(page.getByTestId("preview-a")).toBeVisible();
  expect(demoHits.length).toBe(2);
});

test("E5 fallo 404 en B muestra mensaje sin parciales y conserva stage", async ({ page }) => {
  await page.goto(FRONT);
  const bHits: string[] = [];
  await page.route("**/demo/obama-b.jpg", async (route) => {
    bHits.push(route.request().url());
    await route.fulfill({ status: 404, body: "no" });
  });
  const initialStage = await page.locator("#stage-text").textContent();
  await page.locator(DEMO_BTN).click();
  // Prueba de que el intento ocurrió (si el click fuera no-op, esto falla).
  await expect.poll(async () => bHits.length, { timeout: 10_000 }).toBe(1);
  const btn = page.locator(DEMO_BTN);
  await expect(btn).toBeEnabled({ timeout: 10_000 });
  await expect(btn).toContainText(/Cargar demo/);
  await expect(btn).not.toHaveAttribute("aria-busy", "true");
  // Mensaje exigible: literal único, canal #result-error con role=alert.
  await expect(page.locator("#result-error")).toBeVisible();
  await expect(page.locator("#result-error")).toHaveText(DEMO_LOAD_ERROR_MESSAGE);
  // Sin parciales: inputs vacíos, previews ocultos, stage intacto.
  for (const slot of ["a", "b"] as const) {
    const count = await page.evaluate((s) => {
      const input = document.querySelector<HTMLInputElement>(`input[name="image_${s}"]`);
      return input?.files?.length ?? -1;
    }, slot);
    expect(count).toBe(0);
  }
  await expect(page.getByTestId("preview-a")).toBeHidden();
  await expect(page.getByTestId("preview-b")).toBeHidden();
  await expect(page.locator("#stage-text")).toHaveText(initialStage ?? "");
});

test("E6 fallo demo conserva archivos previos del usuario", async ({ page }) => {
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    Buffer.alloc(56, 0),
  ]);
  await page.goto(FRONT);
  await page
    .locator('input[name="image_a"]')
    .setInputFiles({ name: "mia.png", mimeType: "image/png", buffer: png });
  await expect(page.getByTestId("preview-a")).toBeVisible();
  const initialStage = await page.locator("#stage-text").textContent();
  await page.route("**/demo/obama-b.jpg", async (route) => {
    await route.fulfill({ status: 404, body: "no" });
  });
  await page.locator(DEMO_BTN).click();
  const btn = page.locator(DEMO_BTN);
  await expect(btn).toBeEnabled({ timeout: 10_000 });
  await expect(page.locator("#result-error")).toHaveText(DEMO_LOAD_ERROR_MESSAGE);
  // Estado previo intacto: el archivo del usuario sobrevive al fallo demo.
  const fileName = await page.evaluate(() => {
    const input = document.querySelector<HTMLInputElement>('input[name="image_a"]');
    return input?.files?.[0]?.name ?? null;
  });
  expect(fileName).toBe("mia.png");
  await expect(page.getByTestId("preview-a")).toBeVisible();
  await expect(page.getByTestId("preview-a")).toHaveAttribute("src", /^blob:/);
  await expect(page.getByTestId("preview-b")).toBeHidden();
  await expect(page.locator("#stage-text")).toHaveText(initialStage ?? "");
});

test("E4 tras demo, Comparar envia 2 JPEG y abre job", async ({ page }) => {
  await loadDemo(page);
  await page.getByRole("button", { name: /Comparar/ }).click();
  await expect(page.locator("#stage-text")).toContainText(
    /job .*(cola|enviando|procesando|listo)/,
    { timeout: 15_000 },
  );
});
