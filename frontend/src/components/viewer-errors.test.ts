import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

// Story-178 (S5): errores por fase + carrera vultus:clear + token de generacion.
// TDD slice 1 (RED primero): vocabulario puro en three-rig.ts. Toda
// asercion pasa por la interfaz publica, nunca por internos del visor.

// Given-When-Then por criterio de aceptacion del issue #178:
// 1. fallo en loadAsync -> invalid con causa de parse (fase + cara + name/message)
// 2. fallo renderer/PMREM/render con GLB parseado -> "visor no disponible (WebGL)",
//    nunca "GLB invalido"; se escucha webglcontextlost
// 3. vultus:clear durante loadAsync en vuelo -> carga cancelada, no invalid
// 4. reintento con generacion nueva + meshes viejo -> se ignora (token gen)
// 5. test headless clear mid-load -> cancelacion (ver e2e/viewer-errors.spec.ts)
import {
  describeErrorCause,
  isStaleGen,
  parseGenToken,
  parseViewerStatus,
  viewerCancelledMessage,
  viewerErrorStatus,
  viewerParseErrorMessage,
  viewerStatusMessage,
  viewerWebglErrorMessage,
} from "./three-rig.ts";

describe("S5: vocabulario de error por fase (seam three-rig)", () => {
  it("Given estado webgl roto, When parseo status, Then acepta unavailable y cancelled", () => {
    expect(parseViewerStatus("unavailable")).toEqual({ ok: true, value: "unavailable" });
    expect(parseViewerStatus("cancelled")).toEqual({ ok: true, value: "cancelled" });
    // Vocabulario cerrado: nada fuera del union pasa.
    expect(parseViewerStatus("error").ok).toBe(false);
    expect(parseViewerStatus("").ok).toBe(false);
    expect(parseViewerStatus(null).ok).toBe(false);
    expect(() => parseViewerStatus({})).not.toThrow();
  });

  it("Given GLB parseado pero WebGL roto, When mensaje, Then visor no disponible sin mencionar zip invalido", () => {
    const msg = viewerStatusMessage("unavailable");
    expect(msg).toMatch(/visor no disponible \(WebGL\)/);
    expect(msg.includes("GLB invalido")).toBe(false);
    expect(msg.includes("revisa el zip")).toBe(false);
  });

  it("Given clear a mitad de carga, When mensaje, Then cancelada sin invalid ni revisa-el-zip", () => {
    const msg = viewerStatusMessage("cancelled");
    expect(msg).toMatch(/cancelada/);
    expect(msg.toLowerCase().includes("invalido")).toBe(false);
    expect(msg.includes("revisa el zip")).toBe(false);
    expect(viewerCancelledMessage()).toBe(msg);
  });

  it("Given fallo de parse, When mensaje con causa, Then fase + cara + name/message y revisa-el-zip", () => {
    const msg = viewerParseErrorMessage("A", { name: "SyntaxError", message: "fin inesperado" });
    expect(msg).toMatch(/GLB invalido/);
    expect(msg).toMatch(/cara A/);
    expect(msg).toMatch(/fase parse/);
    expect(msg.includes("SyntaxError")).toBe(true);
    expect(msg.includes("fin inesperado")).toBe(true);
    expect(msg.includes("revisa el zip")).toBe(true);
  });

  it("Given fallo WebGL con GLB parseado, When mensaje con causa, Then fase + cara sin invalid", () => {
    const msg = viewerWebglErrorMessage("B", "render", { name: "Error", message: "context lost" });
    expect(msg).toMatch(/visor no disponible \(WebGL\)/);
    expect(msg).toMatch(/cara B/);
    expect(msg).toMatch(/fase render/);
    expect(msg.includes("context lost")).toBe(true);
    expect(msg.includes("GLB invalido")).toBe(false);
    expect(msg.includes("revisa el zip")).toBe(false);
  });

  it("Given causa desconocida, When describo, Then name/message totales sin lanzar", () => {
    expect(describeErrorCause(new TypeError("fetch failed"))).toEqual({
      name: "TypeError",
      message: "fetch failed",
    });
    const fromString = describeErrorCause("boom");
    expect(fromString.name.length).toBeGreaterThan(0);
    expect(fromString.message.includes("boom")).toBe(true);
    expect(() => describeErrorCause(null)).not.toThrow();
    expect(() => describeErrorCause(undefined)).not.toThrow();
    expect(() => describeErrorCause(Symbol())).not.toThrow();
  });

  it("Given detail con gen, When parseo token, Then entero finito >= 0 o null, nunca throw", () => {
    expect(parseGenToken(0)).toBe(0);
    expect(parseGenToken(7)).toBe(7);
    expect(parseGenToken(-1)).toBeNull();
    expect(parseGenToken(1.5)).toBeNull();
    expect(parseGenToken(Number.NaN)).toBeNull();
    expect(parseGenToken("3")).toBeNull();
    expect(parseGenToken(undefined)).toBeNull();
    expect(parseGenToken(null)).toBeNull();
    expect(() => parseGenToken(Symbol())).not.toThrow();
  });

  it("Given reintento con gen nueva, When llega meshes viejo, Then es stale y el nuevo no", () => {
    expect(isStaleGen(5, 7)).toBe(true);
    expect(isStaleGen(7, 7)).toBe(false);
    expect(isStaleGen(8, 7)).toBe(false);
    // Sin gen vigente aun, nada es stale: primer resultado siempre entra.
    expect(isStaleGen(5, null)).toBe(false);
    expect(isStaleGen(null, 7)).toBe(false);
  });

  it("Given error taxonomico, When mapeo a estado, Then Parse->invalid, WebGL->unavailable, Cancelled->cancelled, Stale->null", () => {
    expect(
      viewerErrorStatus({
        kind: "ParseError",
        face: "A",
        phase: "parse",
        name: "Error",
        message: "x",
      }),
    ).toBe("invalid");
    expect(
      viewerErrorStatus({
        kind: "WebGLError",
        face: "B",
        phase: "render",
        name: "Error",
        message: "y",
      }),
    ).toBe("unavailable");
    expect(viewerErrorStatus({ kind: "CancelledError" })).toBe("cancelled");
    // Stale nunca toca el status: el visor ignora el evento viejo en silencio.
    expect(viewerErrorStatus({ kind: "StaleLoad" })).toBeNull();
  });

  it("Given zip-8 intacto, When mensaje invalid base, Then sigue diciendo GLB invalido + revisa el zip", () => {
    expect(viewerStatusMessage("invalid")).toMatch(/GLB invalido/);
    expect(viewerStatusMessage("invalid").includes("revisa el zip")).toBe(true);
  });
});

describe("S5: cableado del visor (snapshots de fuente ThreeViewer + index)", () => {
  const viewerSrc = readFileSync(new URL("./ThreeViewer.astro", import.meta.url), "utf8");
  const indexSrc = readFileSync(new URL("../pages/index.astro", import.meta.url), "utf8");

  it("Given GLB parseado y contexto perdido, When escucho canvas, Then hay listener webglcontextlost", () => {
    expect(viewerSrc.includes("webglcontextlost")).toBe(true);
  });

  it("Given fallo por fase, When pinto caras, Then taxonomia Parse/WebGL/Cancelled sin catch unico a invalid", () => {
    expect(viewerSrc.includes("ParseError")).toBe(true);
    expect(viewerSrc.includes("WebGLError")).toBe(true);
    expect(viewerSrc.includes("CancelledError")).toBe(true);
    expect(viewerSrc.includes("viewerParseErrorMessage")).toBe(true);
    expect(viewerSrc.includes("viewerWebglErrorMessage")).toBe(true);
  });

  it("Given clear en vuelo, When falla el fetch revocado, Then se detecta cancelacion y no invalid", () => {
    expect(viewerSrc.includes("StaleLoad")).toBe(true);
    expect(viewerSrc.includes("viewerCancelledMessage")).toBe(true);
    // El guard de invalidacion distingue clear (cancelled) de reintento (stale).
    expect(viewerSrc.includes("invalidationSerial")).toBe(true);
  });

  it("Given reintento con gen nueva, When llega meshes viejo, Then token gen lo ignora", () => {
    expect(viewerSrc.includes("parseGenToken")).toBe(true);
    expect(viewerSrc.includes("isStaleGen")).toBe(true);
    expect(indexSrc.includes("detail: { meshA, meshB, pbrA, pbrB, gen: myGen }")).toBe(true);
  });
});
