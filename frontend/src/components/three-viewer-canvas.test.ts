import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

function readViewer(): string {
  return readFileSync(new URL("./ThreeViewer.astro", import.meta.url), "utf8");
}

// TDD RED: Story-177 — el canvas WebGL nunca debe recibir getContext("2d").
// Tracer bullet a nivel seam publica (fuente Astro): el neutro 2D vive en
// overlay dedicado viewer-neutral-a/b; paintFace oculta el overlay antes
// del WebGLRenderer; el fallback skinMaterial se preserva.
describe("Story-177: canvas WebGL dedicado sin contexto 2D previo", () => {
  it("Given boot sin resultado, When neutro visible, Then WebGL nunca recibe getContext 2d", () => {
    const src = readViewer();
    // Overlays dedicados existen en HTML estatico.
    expect(src.includes("viewer-neutral-a")).toBe(true);
    expect(src.includes("viewer-neutral-b")).toBe(true);
    // renderNeutral solo se invoca con overlays neutros, nunca con canvasA/B WebGL.
    expect(src.includes("renderNeutral(neutralA)")).toBe(true);
    expect(src.includes("renderNeutral(neutralB)")).toBe(true);
    expect(src.includes("renderNeutral(canvasA)")).toBe(false);
    expect(src.includes("renderNeutral(canvasB)")).toBe(false);
  });

  it("Given vultus:meshes tras done, When paintFace crea renderer, Then no hay 2d previo y A+B renderizan", () => {
    const src = readViewer();
    // El unico getContext("2d") ejecutable del archivo vive en renderNeutral.
    const hits = (src.match(/\.getContext\("2d"\)/g) ?? []).length;
    expect(hits).toBe(1);
    const idx2d = src.indexOf('canvas.getContext("2d")');
    const idxNeutralFn = src.indexOf("function renderNeutral");
    expect(idx2d).toBeGreaterThan(idxNeutralFn);
    // paintFace oculta el overlay neutro y luego crea el renderer WebGL.
    expect(src.includes("WebGLRenderer")).toBe(true);
    expect(src.includes(".hidden = true")).toBe(true);
    // Ambas caras se pintan en paralelo.
    expect(src.includes("paintFace(THREE, canvasA")).toBe(true);
    expect(src.includes("paintFace(THREE, canvasB")).toBe(true);
  });

  it("Given GLB minimo sin material, When carga, Then fallback skinMaterial sin romper ojos", () => {
    const src = readViewer();
    expect(src.includes("if (current) return")).toBe(true);
    expect(src.includes("fallback.map = map")).toBe(true);
    expect(src.includes("resolveSkinColorHex(true)")).toBe(true);
    expect(src.includes("node.name")).toBe(false);
  });

  it("Sin cambio de contrato zip-8 ni ZIP_MANIFEST", () => {
    const src = readViewer();
    expect(src.includes("ZIP_MANIFEST")).toBe(false);
    expect(src.toLowerCase().includes("zip-8")).toBe(false);
  });
});
