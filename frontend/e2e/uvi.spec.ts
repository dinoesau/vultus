import { test, expect } from "@playwright/test";

// Story-156 Uvi: tamagotchi de espera debajo de hero-card-actions.
// E2E sin backend (rutas mockeadas); solo DOM + CSS + cableado.
const FRONT = process.env.FRONT_URL ?? "http://localhost:4321";

test("156 AC1 idle sin job ni demo: Uvi oculto", async ({ page }) => {
  await page.goto(FRONT);
  const uvi = page.getByTestId("uvi");
  await expect(uvi).toBeHidden();
  await expect(uvi).toHaveAttribute("aria-hidden", "true");
});

test("156 AC2 demo load: Uvi aparece durante carga y se oculta al terminar", async ({
  page,
}) => {
  await page.goto(FRONT);
  let releaseGate: () => void = () => {};
  const gate = new Promise<void>((resolve) => {
    releaseGate = resolve;
  });
  await page.route("**/demo/obama-*.jpg", async (route) => {
    await gate;
    await route.continue();
  });
  await page.locator("#btn-demo").click();
  // Visible mientras el par demo vuela (modo demo: sin scanline job).
  await expect(page.getByTestId("uvi")).toBeVisible();
  await expect(page.locator("#uvi.uvi--job")).toHaveCount(0);
  releaseGate();
  await expect(page.getByTestId("preview-a")).toBeVisible({ timeout: 10_000 });
  // Demo lista sin job: vuelve a idle oculto.
  await expect(page.getByTestId("uvi")).toBeHidden({ timeout: 10_000 });
});

test("156 AC2 click Comparar: Uvi aparece en modo job hasta terminar", async ({
  page,
}) => {
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    Buffer.alloc(56, 0),
  ]);
  await page.goto(FRONT);
  await page.route("**/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ contract_version: 3 }),
    });
  });
  let releaseCompare: () => void = () => {};
  const compareGate = new Promise<void>((resolve) => {
    releaseCompare = resolve;
  });
  await page.route("**/v1/compare", async (route) => {
    await compareGate;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ job_id: "uvi-test-1" }),
    });
  });
  // WS real caido -> fallback polling: processing eterno, Uvi sigue visible.
  await page.route("**/v1/jobs/uvi-test-1", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "processing" }),
    });
  });
  await page
    .locator('input[name="image_a"]')
    .setInputFiles({ name: "a.png", mimeType: "image/png", buffer: png });
  await page
    .locator('input[name="image_b"]')
    .setInputFiles({ name: "b.png", mimeType: "image/png", buffer: png });
  await page.getByRole("button", { name: /Comparar/ }).click();
  // Aparece al enviar (modo job con scanline) aunque el POST siga gated.
  await expect(page.getByTestId("uvi")).toBeVisible({ timeout: 10_000 });
  await expect(page.locator("#uvi.uvi--job")).toHaveCount(1);
  releaseCompare();
  // Polling processing (sin terminal): sigue visible con scanline.
  await expect(page.locator("#stage-text")).toContainText(/uvi-test-1/, {
    timeout: 15_000,
  });
  await expect(page.getByTestId("uvi")).toBeVisible({ timeout: 10_000 });
});

test("156 AC3 prefers-reduced-motion: Uvi estatico sin animacion", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(FRONT);
  const reduced = await page.evaluate(() =>
    window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  expect(reduced).toBe(true);
  const animations = await page.evaluate(() => {
    const pick = (sel: string) => {
      const el = document.querySelector(sel);
      if (!(el instanceof Element)) return "missing";
      return getComputedStyle(el).animationName;
    };
    return {
      body: pick("#uvi .uvi-body"),
      eye: pick("#uvi .uvi-eye"),
      dot: pick("#uvi .uvi-antenna-dot"),
    };
  });
  expect(animations.body).toBe("none");
  expect(animations.eye).toBe("none");
  expect(animations.dot).toBe("none");
});

test("156 AC3 sin reduced-motion: animaciones vivas (breathe/blink/poll)", async ({
  page,
}) => {
  await page.goto(FRONT);
  const animations = await page.evaluate(() => {
    const pick = (sel: string) => {
      const el = document.querySelector(sel);
      if (!(el instanceof Element)) return "missing";
      return getComputedStyle(el).animationName;
    };
    return {
      body: pick("#uvi .uvi-body"),
      eye: pick("#uvi .uvi-eye"),
      dot: pick("#uvi .uvi-antenna-dot"),
    };
  });
  expect(animations.body).toContain("uvi-breathe");
  expect(animations.eye).toContain("uvi-blink");
  expect(animations.dot).toContain("uvi-poll");
});

test("156 AC4 Uvi 72px centrado, SVG inline, cero requests de imagen", async ({
  page,
}) => {
  const images: string[] = [];
  page.on("request", (req) => {
    if (req.resourceType() === "image" && !req.url().includes("favicon")) {
      images.push(req.url());
    }
  });
  await page.goto(FRONT);
  const uvi = page.getByTestId("uvi");
  // Debajo de acciones, dentro del hero-card.
  const actionsBox = await page.locator("#intake .hero-card-actions").boundingBox();
  const uviBox = await uvi.boundingBox();
  // Hidden => sin caja; se afirma placement via DOM order en su lugar.
  expect(uviBox).toBeNull();
  const order = await page.evaluate(() => {
    const actions = document.querySelector("#intake .hero-card-actions");
    const node = document.getElementById("uvi");
    const card = document.getElementById("intake");
    if (!(actions instanceof Element) || !(node instanceof Element)) return null;
    const pos = (el: Element) => {
      let n = 0;
      let cur: Node | null = el;
      const root = document.body;
      const it = root.ownerDocument.createNodeIterator(root, NodeFilter.SHOW_ELEMENT);
      let found: Node | null;
      while ((found = it.nextNode())) {
        if (found === cur) return n;
        n += 1;
      }
      return -1;
    };
    void card;
    return {
      actions: pos(actions),
      uvi: pos(node),
      inCard: card?.contains(node) ?? false,
    };
  });
  expect(order).not.toBeNull();
  expect(order!.uvi).toBeGreaterThan(order!.actions);
  expect(order!.inCard).toBe(true);
  expect(actionsBox).not.toBeNull();
  // SVG inline sin <img>; 72px con margenes auto (centrado).
  await expect(page.locator("#uvi svg.uvi-stage")).toHaveCount(1);
  await expect(page.locator("#uvi img")).toHaveCount(0);
  const metrics = await page.evaluate(() => {
    const node = document.getElementById("uvi");
    const stage = document.querySelector("#uvi .uvi-stage");
    if (!(node instanceof HTMLElement) || !(stage instanceof Element)) return null;
    const cs = getComputedStyle(node);
    const stageBox = (stage as unknown as SVGSVGElement).viewBox.baseVal;
    return {
      width: cs.width,
      marginLeft: cs.marginLeft,
      marginRight: cs.marginRight,
      viewBox: `${stageBox.x} ${stageBox.y} ${stageBox.width} ${stageBox.height}`,
    };
  });
  expect(metrics).not.toBeNull();
  expect(metrics!.width).toBe("72px");
  expect(metrics!.marginLeft).toBe(metrics!.marginRight);
  expect(images).toEqual([]);
});
