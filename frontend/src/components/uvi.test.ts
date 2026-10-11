import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import {
  UVI_BUSY_CLASS,
  UVI_ID,
  UVI_TESTID,
  hideUvi,
  isUviVisible,
  parseUviElement,
  showUvi,
} from "./uvi.ts";

// AC1: idle sin job ni demo -> Uvi oculto (hidden).
// AC2: click demo/Comparar -> Uvi aparece debajo de acciones.
// AC3: prefers-reduced-motion -> estatico sin animacion.
// AC4: cero requests imagen, todo CSS/SVG inline.
//
// Seam bajo prueba: modulo uvi.ts (parse/show/hide totales) + snapshot de
// index.astro (placement + CSS + cableado). Sin jsdom: fakes minimos de DOM.
// Node no trae HTMLDivElement: se registra un FakeDiv global para que el
// `instanceof` de produccion (correcto en navegador) funcione en el test.
class FakeDiv {
  hidden = true;
  readonly _attrs = new Map<string, string>([["aria-hidden", "true"]]);
  readonly _classes = new Set<string>();
  readonly classList = {
    add: (c: string): void => {
      this._classes.add(c);
    },
    remove: (c: string): void => {
      this._classes.delete(c);
    },
    contains: (c: string): boolean => this._classes.has(c),
  };

  setAttribute(k: string, v: string): void {
    this._attrs.set(k, v);
  }

  getAttribute(k: string): string | null {
    return this._attrs.get(k) ?? null;
  }
}

const g = globalThis as unknown as Record<string, unknown>;
if (typeof g["HTMLDivElement"] === "undefined") g["HTMLDivElement"] = FakeDiv;

function makeFakeEl(): {
  el: HTMLDivElement;
  classes: Set<string>;
} {
  const el = new FakeDiv() as unknown as HTMLDivElement;
  const classes = (el as unknown as FakeDiv)._classes;
  return { el, classes };
}

function readIndex(): string {
  return readFileSync(new URL("../pages/index.astro", import.meta.url), "utf8");
}

describe("uvi: AC1 idle oculto, show/hide totales sin lanzar", () => {
  it("hideUvi deja hidden + aria-hidden true sin busy", () => {
    const { el, classes } = makeFakeEl();
    el.hidden = false;
    classes.add(UVI_BUSY_CLASS);
    hideUvi(el);
    expect(el.hidden).toBe(true);
    expect(el.getAttribute("aria-hidden")).toBe("true");
    expect(classes.has(UVI_BUSY_CLASS)).toBe(false);
    expect(isUviVisible(el)).toBe(false);
  });

  it("showUvi demo quita hidden sin scanline (sin busy)", () => {
    const { el, classes } = makeFakeEl();
    showUvi(el, "demo");
    expect(el.hidden).toBe(false);
    expect(el.getAttribute("aria-hidden")).toBe("false");
    expect(classes.has(UVI_BUSY_CLASS)).toBe(false);
    expect(isUviVisible(el)).toBe(true);
  });

  it("showUvi job quita hidden con scanline (busy)", () => {
    const { el, classes } = makeFakeEl();
    showUvi(el, "job");
    expect(el.hidden).toBe(false);
    expect(classes.has(UVI_BUSY_CLASS)).toBe(true);
    expect(isUviVisible(el)).toBe(true);
  });

  it("parseUviElement: ausente y nodo erróneo son Err, nunca lanza", () => {
    const missing = {
      querySelector: () => null,
    } as unknown as ParentNode;
    expect(parseUviElement(missing).ok).toBe(false);
    const wrong = {
      querySelector: () => ({}) as unknown as Element,
    } as unknown as ParentNode;
    const parsedWrong = parseUviElement(wrong);
    expect(parsedWrong.ok).toBe(false);
    expect(() => parseUviElement(missing)).not.toThrow();
  });
});

describe("uvi: AC2 placement debajo de hero-card-actions + cableado", () => {
  it("index tiene #uvi hidden con testid debajo de hero-card-actions en hero-card", () => {
    const src = readIndex();
    expect(src.includes(`id="${UVI_ID}"`)).toBe(true);
    expect(src.includes(`data-testid="${UVI_TESTID}"`)).toBe(true);
    // Hidden por defecto (AC1).
    const uviTag = src.slice(src.indexOf(`id="${UVI_ID}"`) - 40, src.indexOf(`id="${UVI_ID}"`) + 120);
    expect(uviTag.includes("hidden")).toBe(true);
    // Placement: acciones antes que uvi antes que cierre de hero-card.
    const actionsAt = src.indexOf("hero-card-actions");
    const uviAt = src.indexOf(`id="${UVI_ID}"`);
    const intakeClose = src.indexOf('id="demo-credit"');
    expect(actionsAt).toBeGreaterThan(-1);
    expect(uviAt).toBeGreaterThan(actionsAt);
    // Uvi vive dentro del card (antes del strip que sigue al hero).
    expect(src.indexOf('<div class="strip">')).toBeGreaterThan(uviAt);
    void intakeClose;
  });

  it("index cablea show en loadDemo y startJob, hide en terminal (sin duplicar literales)", () => {
    const src = readIndex();
    // Fuente única: importa el seam, no repite "uvi"/"uvi--job" inline.
    expect(src.includes("from \"../components/uvi.ts\"")).toBe(true);
    expect(src.includes("showUvi")).toBe(true);
    expect(src.includes("hideUvi")).toBe(true);
    expect(src.includes("UVI_BUSY_CLASS")).toBe(false);
    // Show en inicio demo + inicio job; hide en done/failed/render/fetch.
    // Los helpers se llaman uviShow(/uviHide(: el import usa showUvi/hideUvi.
    const showCount = src.split("uviShow(").length - 1;
    expect(showCount).toBeGreaterThanOrEqual(3);
    const hideCount = src.split("uviHide(").length - 1;
    expect(hideCount).toBeGreaterThanOrEqual(3);
  });
});

describe("uvi: AC3 reduced-motion estatico + AC4 cero imagen", () => {
  it("CSS trae marcha pixel, leds, poll y scanline job con media query que apaga todo", () => {
    const src = readIndex();
    expect(src.includes("uvi-march")).toBe(true);
    expect(src.includes("steps(")).toBe(true);
    expect(src.includes("uvi-led")).toBe(true);
    expect(src.includes("uvi-poll")).toBe(true);
    expect(src.includes(UVI_BUSY_CLASS)).toBe(true);
    expect(src.includes("uvi-scan")).toBe(true);
    expect(src.includes("prefers-reduced-motion")).toBe(true);
    expect(src.includes("animation: none")).toBe(true);
  });

  it("Pixel Walker es SVG pixelado inline 72px centrado, sin <img> ni url( ni http", () => {
    const src = readIndex();
    const uviAt = src.indexOf(`id="${UVI_ID}"`);
    expect(uviAt).toBeGreaterThan(-1);
    // Bloque Uvi: desde el div hasta el cierre del caption.
    const tail = src.slice(uviAt, uviAt + 6000);
    expect(tail.includes("<svg")).toBe(true);
    expect(tail.includes("</svg>")).toBe(true);
    expect(tail.includes("<img")).toBe(false);
    // Cero url(): sin clip-path ni assets, todo rects inline.
    expect(tail.includes("url(")).toBe(false);
    expect(tail.includes("http")).toBe(false);
    // Grid pixel 16x16 con bordes duros.
    expect(tail.includes('viewBox="0 0 16 16"')).toBe(true);
    expect(src.includes("crispEdges")).toBe(true);
    expect(src.includes("pixelated")).toBe(true);
    // 72px + centrado.
    expect(src.includes("72px")).toBe(true);
    expect(src.includes("margin") && src.includes("auto")).toBe(true);
  });

  it("contenedor vacio para frases (siguiente worker) sin logica de rotacion", () => {
    const src = readIndex();
    expect(src.includes('id="uvi-text"')).toBe(true);
    expect(src.includes("uvi-phrase")).toBe(false);
    // uvi-text solo en CSS (#uvi-text x2) + el <p> vacio: ningun JS lo
    // referencia, asi que no hay timer ni rotacion cableada.
    expect(src.split("uvi-text").length - 1).toBe(3);
    expect(src.includes('getElementById("uvi-text")')).toBe(false);
  });
});
