import { test, expect } from "@playwright/test";

// Story-158: Ver estudio vive solo en right panel (hero-card), debajo de
// Uvi/frases, encima de #demo-credit. Desaparece del hero izquierdo.
// Given-When-Then como acceptance del issue.
const FRONT = process.env.FRONT_URL ?? "http://localhost:4321";

test("158 Given hero izquierdo, When miro btnrow, Then solo Comparar ahora", async ({
  page,
}) => {
  await page.goto(FRONT);
  const btnrow = page.locator(".hero .btnrow");
  await expect(
    btnrow.getByRole("link", { name: /Comparar ahora/ }),
  ).toBeVisible();
  await expect(
    btnrow.getByRole("link", { name: /Ver estudio/ }),
  ).toHaveCount(0);
});

test("158 Given right panel, When miro debajo de frases, Then Ver estudio full-width a #estudio", async ({
  page,
}) => {
  await page.goto(FRONT);
  const btn = page.getByTestId("btn-estudio");
  await expect(btn).toBeVisible();
  await expect(btn).toHaveAttribute("href", "#estudio");
  await expect(btn).toHaveText(/Ver estudio/);
  const order = await page.evaluate(() => {
    const uvi = document.getElementById("uvi");
    const text = document.getElementById("uvi-text");
    const btnEl = document.getElementById("btn-estudio");
    const credit = document.getElementById("demo-credit");
    const card = document.getElementById("intake");
    if (!(uvi instanceof Element) || !(btnEl instanceof Element)) return null;
    const pos = (el: Element) => {
      let n = 0;
      const it = document.body.ownerDocument.createNodeIterator(
        document.body,
        NodeFilter.SHOW_ELEMENT,
      );
      let found: Node | null;
      while ((found = it.nextNode())) {
        if (found === el) return n;
        n += 1;
      }
      return -1;
    };
    return {
      uvi: pos(uvi),
      text: text instanceof Element ? pos(text) : -1,
      btn: pos(btnEl),
      credit: credit instanceof Element ? pos(credit) : -1,
      inCard: card?.contains(btnEl) ?? false,
    };
  });
  expect(order).not.toBeNull();
  expect(order!.btn).toBeGreaterThan(order!.uvi);
  expect(order!.btn).toBeGreaterThan(order!.text);
  expect(order!.credit).toBeGreaterThan(order!.btn);
  expect(order!.inCard).toBe(true);
  // Ghost full-width: ocupa todo el ancho del form, centrado.
  const metrics = await page.evaluate(() => {
    const btnEl = document.getElementById("btn-estudio");
    const form = document.getElementById("compare");
    if (!(btnEl instanceof HTMLElement) || !(form instanceof HTMLElement))
      return null;
    const b = btnEl.getBoundingClientRect();
    const f = form.getBoundingClientRect();
    const cs = getComputedStyle(btnEl);
    return {
      btnW: b.width,
      formW: f.width,
      justify: cs.justifyContent,
      marginTop: cs.marginTop,
    };
  });
  expect(metrics).not.toBeNull();
  expect(Math.abs(metrics!.btnW - metrics!.formW)).toBeLessThan(48);
  expect(metrics!.justify).toBe("center");
});

test("158 Given click en Ver estudio, When navego, Then scroll smooth a #estudio", async ({
  page,
}) => {
  await page.goto(FRONT);
  const smooth = await page.evaluate(
    () => getComputedStyle(document.documentElement).scrollBehavior,
  );
  expect(smooth).toBe("smooth");
  await page.getByTestId("btn-estudio").click();
  await expect(page.locator("#estudio")).toBeInViewport();
  expect(page.url()).toContain("#estudio");
});

test("158 Given Uvi oculto, When miro right panel, Then Ver estudio sigue visible", async ({
  page,
}) => {
  await page.goto(FRONT);
  await expect(page.getByTestId("uvi")).toBeHidden();
  await expect(page.getByTestId("btn-estudio")).toBeVisible();
});

test("158 movil: boton debajo del card sin romper dropgrid 1fr", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(FRONT);
  const btn = page.getByTestId("btn-estudio");
  await expect(btn).toBeVisible();
  const cols = await page.evaluate(() => {
    const grid = document.querySelector("#intake .dropgrid");
    if (!(grid instanceof HTMLElement)) return null;
    return getComputedStyle(grid).gridTemplateColumns;
  });
  expect(cols).not.toBeNull();
  // 1fr unico en movil: una sola columna.
  expect(cols!.trim().split(" ").length).toBe(1);
  const stacked = await page.evaluate(() => {
    const btnEl = document.getElementById("btn-estudio");
    const card = document.getElementById("intake");
    if (!(btnEl instanceof HTMLElement) || !(card instanceof HTMLElement))
      return null;
    const b = btnEl.getBoundingClientRect();
    const c = card.getBoundingClientRect();
    return { insideH: b.left >= c.left - 2 && b.right <= c.right + 2 };
  });
  expect(stacked?.insideH).toBe(true);
});
