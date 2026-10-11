import { test, expect, type Page } from "@playwright/test";

// Story-153: solo dot en pantallas chicas (<=960px).
// E2E sin backend (ruta /health mockeada); solo DOM + CSS + cableado.
// Given-When-Then como acceptance del issue.
const FRONT = process.env.FRONT_URL ?? "http://localhost:4321";

async function mockHealth(page: Page, status: number, body: unknown): Promise<void> {
  await page.route("**/health", async (route) => {
    await route.fulfill({
      status,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
}

test("153 AC1 movil 390px: pill visible con solo el dot, sin texto Health", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await mockHealth(page, 200, { contract_version: 3 });
  await page.goto(FRONT);
  const pill = page.locator("#health-pill");
  await expect(pill).toBeVisible();
  await expect(pill.locator(".health-label")).toBeHidden();
  await expect(pill.locator(".health-dot")).toBeVisible();
  // Sin texto visible: el pill colapsado cabe holgado y no se sale del nav.
  const box = await pill.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.width).toBeLessThan(48);
  expect(box!.x + box!.width).toBeLessThanOrEqual(390);
});

test("153 AC1 frontera: dot-only en 960px, texto completo en 961px", async ({
  page,
}) => {
  await mockHealth(page, 200, { contract_version: 3 });
  await page.setViewportSize({ width: 960, height: 800 });
  await page.goto(FRONT);
  await expect(page.locator("#health-pill")).toBeVisible();
  await expect(page.locator("#health-pill .health-label")).toBeHidden();
  await page.setViewportSize({ width: 961, height: 800 });
  await expect(page.locator("#health-pill .health-label")).toBeVisible();
});

test("153 AC2 movil: lector escucha Health: ok o Health: down", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await mockHealth(page, 200, { contract_version: 3 });
  await page.goto(FRONT);
  await expect(page.locator("#health-pill")).toHaveAttribute("aria-label", "Health: ok");
  await mockHealth(page, 500, {});
  await page.reload();
  await expect(page.locator("#health-pill")).toHaveAttribute(
    "aria-label",
    "Health: down",
  );
});

test("153 AC3 movil: mismo tooltip de desktop via title", async ({ page }) => {
  await mockHealth(page, 200, { contract_version: 3 });
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto(FRONT);
  const desktopTitle = await page.locator("#health-pill").getAttribute("title");
  expect(desktopTitle).toBe("contract v3");
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator("#health-pill .health-label")).toBeHidden();
  await expect(page.locator("#health-pill")).toHaveAttribute("title", desktopTitle!);
});
