import { test, expect } from "@playwright/test";

// Story-178 (S5): errores por fase + carrera vultus:clear + token de gen.
// Headless contra el front servido: dispara eventos DOM reales y aserta
// el status del visor. Given-When-Then como acceptance del issue #178.
const FRONT = process.env.FRONT_URL ?? "http://localhost:4321";

test("178 Given loadAsync en vuelo, When vultus:clear revoca URLs, Then cancelada y nunca invalid", async ({
  page,
}) => {
  await page.goto(FRONT);
  const status = page.locator("#viewer-3d-status");
  await expect(status).toContainText(/neutro/);
  // Blobs que fallan en parse + clear en el mismo task: el fetch del blob
  // revocado resuelve tras el clear y debe clasificarse como cancelada.
  await page.evaluate(() => {
    const blob = (s: string) => new Blob([s], { type: "model/gltf-binary" });
    window.dispatchEvent(
      new CustomEvent("vultus:meshes", {
        detail: { meshA: blob("no-es-un-glb-a"), meshB: blob("no-es-un-glb-b"), gen: 1 },
      }),
    );
    window.dispatchEvent(new Event("vultus:clear"));
  });
  await expect(status).toContainText(/cancelada/, { timeout: 15000 });
  await expect(status).not.toContainText(/GLB invalido/);
  await expect(status).not.toContainText(/revisa el zip/);
});

test("178 Given reintento con gen nueva, When llega meshes viejo, Then el visor lo ignora", async ({
  page,
}) => {
  await page.goto(FRONT);
  // Observable sincrono: el evento stale retorna antes de recrear URLs,
  // asi que el href expuesto por la gen vigente no cambia.
  const seen = await page.evaluate(() => {
    const linkA = document.getElementById("download-mesh-a");
    const blob = (s: string) => new Blob([s], { type: "model/gltf-binary" });
    window.dispatchEvent(
      new CustomEvent("vultus:meshes", {
        detail: { meshA: blob("nueva-a"), meshB: blob("nueva-b"), gen: 5 },
      }),
    );
    const hrefFresh = linkA instanceof HTMLAnchorElement ? linkA.href : null;
    window.dispatchEvent(
      new CustomEvent("vultus:meshes", {
        detail: { meshA: blob("vieja-a"), meshB: blob("vieja-b"), gen: 3 },
      }),
    );
    const hrefAfterStale = linkA instanceof HTMLAnchorElement ? linkA.href : null;
    return { hrefFresh, hrefAfterStale };
  });
  expect(seen.hrefFresh).not.toBeNull();
  expect(seen.hrefAfterStale).toBe(seen.hrefFresh);
});

test("178 Given blobs rotos sin clear, When falla loadAsync, Then invalid con fase + cara + causa", async ({
  page,
}) => {
  await page.goto(FRONT);
  const status = page.locator("#viewer-3d-status");
  await page.evaluate(() => {
    const blob = (s: string) => new Blob([s], { type: "model/gltf-binary" });
    window.dispatchEvent(
      new CustomEvent("vultus:meshes", {
        detail: { meshA: blob("roto-a"), meshB: blob("roto-b"), gen: 9 },
      }),
    );
  });
  // Sin clear en vuelo el parse roto si culpa al zip, con fase y cara.
  await expect(status).toContainText(/GLB invalido/, { timeout: 15000 });
  await expect(status).toContainText(/fase parse/);
  await expect(status).toContainText(/cara [AB]/);
});
