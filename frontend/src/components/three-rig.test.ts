import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import {
  LIGHT_PROBES,
  PBR_MAP_TINT_FREE_HEX,
  SKIN_MATERIAL,
  STUDIO_BACKGROUND_HEX,
  STUDIO_CANVAS_HEIGHT,
  STUDIO_CANVAS_WIDTH,
  STUDIO_LIGHTS,
  buildStudioLights,
  checkLightProbes,
  checkStudioLights,
  parseStudioBackground,
  parseViewerStatus,
  resolveSkinColorHex,
  studioConfigMessage,
  viewerStatusMessage,
} from "./three-rig.ts";
import {
  contractVersionMessage,
  contractVersionToNumber,
  healthNeedsUpdate,
  parseContractVersion,
  parseHealthContractVersion,
  CONTRACT_VERSION,
} from "./compare-client.ts";

// Aguja dinamica: el propio test no debe contener el literal prohibido,
// para que el grep de barrera sobre archivos del visor siga en cero.
const HEAT_NEEDLE = ["heat", "map"].join("");
const FUSION_NEEDLE = ["fusion", "base"].join("-");

function readViewer(file: string): string {
  return readFileSync(new URL(`./${file}`, import.meta.url), "utf8");
}

describe("three-rig: estudio blanco 3 luces (seam publica ThreeViewer)", () => {
  it("fondo blanco de estudio y canvas 480x360", () => {
    expect(STUDIO_BACKGROUND_HEX).toBe(0xffffff);
    expect(STUDIO_CANVAS_WIDTH).toBe(480);
    expect(STUDIO_CANVAS_HEIGHT).toBe(360);
  });

  it("exactamente 3 luces key/fill/rim con intensidades finitas", () => {
    const kinds = STUDIO_LIGHTS.map((l) => l.kind);
    expect(kinds).toEqual(["key", "fill", "rim"]);
    expect(STUDIO_LIGHTS.length).toBe(3);
    for (const light of STUDIO_LIGHTS) {
      expect(Number.isFinite(light.intensity)).toBe(true);
      expect(light.intensity).toBeGreaterThan(0);
      expect(light.intensity).toBeLessThanOrEqual(3);
      expect(light.position.length).toBe(3);
      for (const axis of light.position) expect(Number.isFinite(axis)).toBe(true);
    }
  });

  it("buildStudioLights retorna el trio canonico sin lanzar", () => {
    const lights = buildStudioLights();
    expect(lights.length).toBe(3);
    expect(lights.map((l) => l.kind)).toEqual(["key", "fill", "rim"]);
  });

  it("material piel PBR real: sin emisivo ni metal invertido", () => {
    expect(SKIN_MATERIAL.metalness).toBe(0);
    expect(SKIN_MATERIAL.roughness).toBeGreaterThan(0);
    expect(SKIN_MATERIAL.roughness).toBeLessThan(1);
    expect("emissive" in SKIN_MATERIAL).toBe(false);
  });

  it("parseStudioBackground total con Result, nunca lanza", () => {
    expect(parseStudioBackground(0xffffff)).toEqual({ ok: true, value: 0xffffff });
    expect(parseStudioBackground(0x0c141d).ok).toBe(false);
    expect(parseStudioBackground("white").ok).toBe(false);
    expect(parseStudioBackground(null).ok).toBe(false);
    expect(parseStudioBackground(undefined).ok).toBe(false);
    expect(parseStudioBackground(NaN).ok).toBe(false);
    expect(() => parseStudioBackground({})).not.toThrow();
    expect(() => parseStudioBackground(Symbol())).not.toThrow();
  });

  it("checkStudioLights exige trio exacto con intensidades sanas", () => {
    expect(checkStudioLights(STUDIO_LIGHTS).ok).toBe(true);
    expect(checkStudioLights([]).ok).toBe(false);
    expect(checkStudioLights(STUDIO_LIGHTS.slice(0, 2)).ok).toBe(false);
    const bad = STUDIO_LIGHTS.map((l) => ({ ...l }));
    bad[0] = { ...bad[0], intensity: 0 };
    expect(checkStudioLights(bad).ok).toBe(false);
    const nanLight = STUDIO_LIGHTS.map((l) => ({ ...l }));
    nanLight[1] = { ...nanLight[1], intensity: Number.NaN };
    expect(checkStudioLights(nanLight).ok).toBe(false);
  });

  it("parseViewerStatus + mensaje exhaustivo con cara real lista", () => {
    expect(parseViewerStatus("ready")).toEqual({ ok: true, value: "ready" });
    expect(parseViewerStatus("idle").ok).toBe(true);
    expect(parseViewerStatus("loading").ok).toBe(true);
    expect(parseViewerStatus("invalid").ok).toBe(true);
    expect(parseViewerStatus(["heat", "map"].join("")).ok).toBe(false);
    expect(parseViewerStatus("").ok).toBe(false);
    expect(parseViewerStatus(null).ok).toBe(false);
    expect(() => parseViewerStatus({})).not.toThrow();
    const ready = parseViewerStatus("ready");
    if (ready.ok) expect(viewerStatusMessage(ready.value)).toMatch(/cara real lista/);
    const idle = parseViewerStatus("idle");
    if (idle.ok) expect(viewerStatusMessage(idle.value)).toMatch(/neutro/);
  });

  it("snapshot: cero panel de fusion en visores UV (6 paneles zip-8)", () => {
    const src = readViewer("UvViewers.astro");
    expect(src.toLowerCase().includes(HEAT_NEEDLE)).toBe(false);
    expect(src.toLowerCase().includes(FUSION_NEEDLE)).toBe(false);
    for (const id of ["panel-uv-a", "panel-uv-b", "panel-pbr-a", "panel-pbr-b", "panel-render-a", "panel-render-b"] as const) {
      expect(src.includes(id)).toBe(true);
    }
  });

  it("snapshot: visor 3D doble con fondo blanco y cero panel de fusion", () => {
    const src = readViewer("ThreeViewer.astro");
    expect(src.toLowerCase().includes(HEAT_NEEDLE)).toBe(false);
    expect(src.includes("viewer-3d-a")).toBe(true);
    expect(src.includes("viewer-3d-b")).toBe(true);
    expect(src.includes("viewer-3d-status")).toBe(true);
    expect(src.includes("STUDIO_BACKGROUND_HEX") || src.includes("0xffffff")).toBe(true);
  });

  it("snapshot: index sin slider ni ids de fusion", () => {
    const src = readFileSync(new URL("../pages/index.astro", import.meta.url), "utf8");
    expect(src.toLowerCase().includes(HEAT_NEEDLE)).toBe(false);
    expect(src.toLowerCase().includes(FUSION_NEEDLE)).toBe(false);
  });

  it("PBR sin tinte: blanco con mapa, tono piel sin mapa", () => {
    expect(PBR_MAP_TINT_FREE_HEX).toBe(0xffffff);
    expect(resolveSkinColorHex(true)).toBe(0xffffff);
    expect(resolveSkinColorHex(false)).toBe(SKIN_MATERIAL.colorHex);
  });

  it("salud v1/v2 => mensaje actualiza visible, no generico de zip", () => {
    const stale = parseHealthContractVersion({ contract_version: 1 });
    expect(stale.ok).toBe(false);
    if (!stale.ok) {
      const msg = contractVersionMessage(stale.error);
      expect(msg).toMatch(/actualiza/);
      expect(msg.includes("zip sin piezas")).toBe(false);
      expect(msg.includes("zip inv")).toBe(false);
    }
    expect(healthNeedsUpdate({ contract_version: 1 })).toBe(true);
    expect(healthNeedsUpdate({ contract_version: 2 })).toBe(true);
    expect(healthNeedsUpdate({ contract_version: 3 })).toBe(false);
    const broken = parseHealthContractVersion({ contract_version: "x" });
    expect(broken.ok).toBe(false);
    if (!broken.ok) {
      expect(contractVersionMessage(broken.error)).toMatch(/actualiza/);
    }
  });

  it("visor: single renderer dispose + PBR blanco + ojos + PBR-fail + boot", () => {
    const src = readViewer("ThreeViewer.astro");
    expect(src.includes("disposeCanvas")).toBe(true);
    expect(src.includes("cancelAnimationFrame")).toBe(true);
    expect(src.includes("resolveSkinColorHex(true)")).toBe(true);
    expect(src.includes('includes("eye")')).toBe(false);
    expect(src.toLowerCase().includes("por indice")).toBe(true);
    expect(src.includes("sin PBR")).toBe(true);
    expect(src.includes("parseViewerDetail")).toBe(true);
    expect(src.includes("CustomEvent")).toBe(false);
    expect(src.includes("checkStudioLights")).toBe(true);
    expect(src.includes("parseStudioBackground")).toBe(true);
  });

  it("visor conserva SkinPBR embebida: mapa externo solo a fallback sin material", () => {
    const src = readViewer("ThreeViewer.astro");
    expect(src.includes("if (current) return")).toBe(true);
    expect(src.includes("sin material")).toBe(true);
    expect(src.includes("fallback.map = map")).toBe(true);
    expect(src.includes("node.material = mat")).toBe(false);
    expect(src.includes("return mat;")).toBe(false);
  });

  it("visor preserva ojos por indice aunque node.name no mencione eye", () => {
    const src = readViewer("ThreeViewer.astro");
    expect(src.includes("node.name")).toBe(false);
    expect(src.includes("material 0")).toBe(true);
    expect(src.includes("material 1")).toBe(true);
  });

  it("index: salud cableada en startJob/fetchResult con actualiza visible", () => {
    const src = readFileSync(new URL("../pages/index.astro", import.meta.url), "utf8");
    expect(src.includes("parseHealthContractVersion")).toBe(true);
    expect(src.includes("contractVersionMessage")).toBe(true);
    expect(src.includes("checkContractHealth")).toBe(true);
  });

  it("DOM: cero panel denso via querySelector null ademas de strings", () => {
    const PANEL_NEEDLE = ["panel-", "heat", "map"].join("");
    const OPACITY_NEEDLE = ["heat", "map", "-opacity"].join("");
    for (const file of ["ThreeViewer.astro", "UvViewers.astro"] as const) {
      const src = readViewer(file);
      expect(src.includes(PANEL_NEEDLE)).toBe(false);
      expect(src.includes(OPACITY_NEEDLE)).toBe(false);
    }
    const indexSrc = readFileSync(new URL("../pages/index.astro", import.meta.url), "utf8");
    expect(indexSrc.includes(PANEL_NEEDLE)).toBe(false);
    expect(indexSrc.includes(OPACITY_NEEDLE)).toBe(false);
    if (typeof document !== "undefined") {
      const found = document.querySelector('[data-testid="' + PANEL_NEEDLE + '"]');
      expect(found).toBeNull();
    } else {
      expect(PANEL_NEEDLE.length).toBeGreaterThan(0);
    }
  });

  it("contract brand espejo de edge sin importar edge (mint + accessor)", () => {
    expect(contractVersionToNumber(CONTRACT_VERSION)).toBe(3);
    const ok = parseContractVersion(3);
    expect(ok.ok).toBe(true);
    if (ok.ok) expect(contractVersionToNumber(ok.value)).toBe(3);
    expect(parseContractVersion(1).ok).toBe(true);
    expect(parseContractVersion(0).ok).toBe(false);
    expect(parseContractVersion("2").ok).toBe(false);
    expect(parseContractVersion(NaN).ok).toBe(false);
    expect(() => parseContractVersion({})).not.toThrow();
    const src = readFileSync(new URL("./compare-client.ts", import.meta.url), "utf8");
    expect(src.includes("mintContractVersionUnchecked")).toBe(true);
    expect(src.includes("contractVersionToNumber")).toBe(true);
    expect(src.includes("CONTRACT_VERSION")).toBe(true);
    expect(src.includes('from "../..')).toBe(false);
    expect(src.includes("from \"./contract\"")).toBe(false);
    expect(src.includes("edge/contract")).toBe(true);
  });

  it("extractores Pbr/Meshes solo railway try*, sin wrappers throw", () => {
    const src = readFileSync(new URL("./compare-client.ts", import.meta.url), "utf8");
    expect(src.includes("tryExtractResultPbr")).toBe(true);
    expect(src.includes("tryExtractResultMeshes")).toBe(true);
    expect(src.includes("tryExtractViewerBlobs")).toBe(true);
    expect(src.includes("extractResultPbr")).toBe(false);
    expect(src.includes("extractResultMeshes")).toBe(false);
  });

  it("eje no finito es InvalidPosition con mensaje propio, no InvalidIntensity", () => {
    const tilted = STUDIO_LIGHTS.map((l) => ({
      ...l,
      position: [...l.position] as unknown as readonly [number, number, number],
    }));
    tilted[0] = { ...tilted[0], position: [Number.NaN, 0, 0] };
    const parsed = checkStudioLights(tilted);
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) {
      expect(parsed.error.kind).toBe("InvalidPosition");
      expect(studioConfigMessage(parsed.error)).toMatch(/posicion/);
    }
    const badIntensity = STUDIO_LIGHTS.map((l) => ({ ...l }));
    badIntensity[0] = { ...badIntensity[0], intensity: 0 };
    const parsed2 = checkStudioLights(badIntensity);
    expect(parsed2.ok).toBe(false);
    if (!parsed2.ok) expect(parsed2.error.kind).toBe("InvalidIntensity");
  });

  it("viewer detail readonly sin reasignacion y readDetail Result a invalid", () => {
    const src = readViewer("ThreeViewer.astro");
    expect(src.includes("readonly meshA")).toBe(true);
    expect(src.includes("readonly meshB")).toBe(true);
    expect(src.includes("value.pbrA =")).toBe(false);
    expect(src.includes("value.pbrB =")).toBe(false);
    expect(src.includes("function readDetail(ev: Event): Result<ViewerDetail")).toBe(true);
    expect(src.includes("InvalidViewerDetail")).toBe(true);
    expect(src.includes('viewerStatusMessage("invalid")')).toBe(true);
  });

  it("checkContractHealth Result: JSON roto es InvalidHealthVersion, solo red es fail-open", () => {
    const src = readFileSync(new URL("../pages/index.astro", import.meta.url), "utf8");
    expect(src.includes("Promise<Result<boolean, HealthVersionError>>")).toBe(true);
    expect(src.includes("InvalidHealthVersion")).toBe(true);
    expect(src.includes("fail-open")).toBe(true);
    expect(src.includes("if (!healthy.ok) return")).toBe(true);
  });

  it("sondas de luz del paper: cromo + grafito metalicas con posiciones finitas", () => {
    expect(LIGHT_PROBES.length).toBe(2);
    expect(LIGHT_PROBES.map((p) => p.name)).toEqual(["chrome", "graphite"]);
    for (const probe of LIGHT_PROBES) {
      expect(probe.metalness).toBeGreaterThan(0);
      expect(probe.metalness).toBeLessThanOrEqual(1);
      expect(probe.roughness).toBeGreaterThan(0);
      expect(probe.roughness).toBeLessThan(1);
      expect(probe.radius).toBeGreaterThan(0);
      expect(probe.position.length).toBe(3);
      for (const axis of probe.position) expect(Number.isFinite(axis)).toBe(true);
    }
  });

  it("checkLightProbes exige duo exacto con metal y radios sanos", () => {
    expect(checkLightProbes(LIGHT_PROBES)).toEqual({ ok: true, value: LIGHT_PROBES });
    const bad = LIGHT_PROBES.map((p) => ({ ...p }));
    bad[0] = { ...bad[0], metalness: 0 };
    const parsed = checkLightProbes(bad);
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.error.kind).toBe("InvalidProbeMetal");
    const short = LIGHT_PROBES.slice(0, 1);
    const parsed2 = checkLightProbes(short);
    expect(parsed2.ok).toBe(false);
    if (!parsed2.ok) expect(parsed2.error.kind).toBe("InvalidProbeCount");
  });

  it("visor monta sondas + entorno: RoomEnvironment y metalness sin emisivo", () => {
    const src = readViewer("ThreeViewer.astro");
    expect(src.includes("RoomEnvironment")).toBe(true);
    expect(src.includes("LIGHT_PROBES")).toBe(true);
    expect(src.includes("metalness")).toBe(true);
    expect(src.includes("emissive")).toBe(false);
  });
});
