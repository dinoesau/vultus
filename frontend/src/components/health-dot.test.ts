import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

// Story-153 dot-only movil: guardas estructurales sobre index.astro.
// Un test por criterio Given-When-Then; los literales esperados vienen del
// issue #153, no del codigo bajo prueba.
function readIndex(): string {
  return readFileSync(new URL("../pages/index.astro", import.meta.url), "utf8");
}

function media960(src: string): string {
  const start = src.indexOf("@media (max-width: 960px)");
  const next = src.indexOf("@media (max-width: 640px)");
  return src.slice(start, next);
}

function channel(hex: string): number {
  const c = parseInt(hex, 16) / 255;
  return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function luminance(hex: string): number {
  const clean = hex.replace("#", "");
  const [r, g, b] = [clean.slice(0, 2), clean.slice(2, 4), clean.slice(4, 6)];
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

function contrastRatio(fg: string, bg: string): number {
  const hi = Math.max(luminance(fg), luminance(bg));
  const lo = Math.min(luminance(fg), luminance(bg));
  return (hi + 0.05) / (lo + 0.05);
}

function cssVar(src: string, name: string): string {
  const match = src.match(new RegExp(`${name}:\\s*(#[0-9a-fA-F]{6})`));
  if (!match) throw new Error(`falta var ${name} en index.astro`);
  return match[1];
}

describe("health dot-only movil (issue #153)", () => {
  it("AC1 en <=960px el pill sigue visible y solo el texto se oculta", () => {
    const media = media960(readIndex());
    // El pill completo nunca se apaga: sin display:none sobre .health-pill.
    expect(media).not.toMatch(/\.health-pill\s*\{[^}]*display\s*:/);
    // Solo el label se oculta; el dot queda visible.
    expect(media).toMatch(/\.health-label\s*\{[^}]*display\s*:\s*none/);
  });

  it("AC1 el pill colapsa padding sin romper el flex del nav", () => {
    const src = readIndex();
    expect(src).toMatch(/\.health-pill\s*\{[^}]*inline-flex/);
    const media = media960(src);
    expect(media).toMatch(/\.health-pill\s*\{[^}]*min-width\s*:\s*0/);
  });

  it("AC2 el poller existente pinta aria-label Health: ok/down", () => {
    const src = readIndex();
    expect(src).toContain("healthAriaLabel(state.ok)");
    expect(src).toMatch(/aria-label="Health: (ok|down|checking)"/);
  });

  it("AC3 el mismo tooltip de desktop vive en el pill sin ramas por viewport", () => {
    const src = readIndex();
    expect(src).toContain("healthPill.title = state.title;");
    expect(src).not.toMatch(/matchMedia|innerWidth/);
  });

  it("AC4 contraste AA del dot sobre --canvas en ambos colores", () => {
    const src = readIndex();
    const canvas = cssVar(src, "--canvas");
    const okRatio = contrastRatio(cssVar(src, "--health-ok"), canvas);
    const downRatio = contrastRatio(cssVar(src, "--error"), canvas);
    expect(okRatio).toBeGreaterThanOrEqual(4.5);
    expect(downRatio).toBeGreaterThanOrEqual(4.5);
  });
});
