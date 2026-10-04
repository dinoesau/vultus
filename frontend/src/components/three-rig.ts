// Nucleo funcional del visor estudio blanco: fondo + 3 luces + PBR.
// Sin import de three.js aqui: el shell Astro mapea estos valores a THREE.
// Todo parseo en el borde retorna Result; el core nunca lanza por input esperado.

export type Result<T, E> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

type Brand<T, Name extends string> = T & { readonly __brand: Name };
export type StudioBackground = Brand<number, "StudioBackground">;

// Blanco estudio: el unico fondo aceptado tras el corte del panel oscuro.
export const STUDIO_BACKGROUND_HEX = 0xffffff as const;
export const STUDIO_CANVAS_WIDTH = 480 as const;
export const STUDIO_CANVAS_HEIGHT = 360 as const;
export const STUDIO_CAMERA_FOV = 35 as const;

export type StudioLightKind = "key" | "fill" | "rim";

export interface StudioLight {
  readonly kind: StudioLightKind;
  readonly colorHex: number;
  readonly intensity: number;
  readonly position: readonly [number, number, number];
}

// Trio canonico: clave calida frontal, relleno lateral y recorte trasero.
// Intensidades en (0, 3]: volumen visible sin quemar la piel.
export const STUDIO_LIGHTS: readonly StudioLight[] = [
  { kind: "key", colorHex: 0xffffff, intensity: 1.6, position: [2, 3, 4] },
  { kind: "fill", colorHex: 0xfff2e6, intensity: 0.7, position: [-3, 1, 2] },
  { kind: "rim", colorHex: 0xffffff, intensity: 0.9, position: [0, 2, -4] },
] as const;

// Piel PBR real: solo albedo + rugosidad, sin emisivo ni metal invertido.
// Con mapa albedo el color debe ser blanco para no tintar; sin mapa, tono piel.
export const SKIN_MATERIAL = {
  colorHex: 0xd9a27a,
  roughness: 0.55,
  metalness: 0,
} as const;

export type LightProbeName = "chrome" | "graphite";

export interface LightProbe {
  readonly name: LightProbeName;
  readonly metalness: number;
  readonly roughness: number;
  readonly radius: number;
  readonly position: readonly [number, number, number];
}

// Sondas de luz estilo paper FFHQ-UV: esferas que reflejan el estudio
// junto al busto (son del visor, nunca del zip). Cromo espejo a la
// derecha, grafito satinado a la izquierda; el entorno (RoomEnvironment)
// lo monta el shell Astro, aqui solo datos + validacion.
export const LIGHT_PROBES: readonly LightProbe[] = [
  { name: "chrome", metalness: 1, roughness: 0.06, radius: 0.18, position: [1.55, -0.45, 0.4] },
  { name: "graphite", metalness: 0.9, roughness: 0.35, radius: 0.14, position: [-1.65, -0.55, 0.2] },
] as const;

// Blanco sin tinte cuando hay mapa: el albedo manda, el color no multiplica.
export const PBR_MAP_TINT_FREE_HEX = 0xffffff as const;

/** Color de piel resuelto: blanco con mapa, tono piel sin mapa. Total, sin throw. */
export function resolveSkinColorHex(hasMap: boolean): number {
  return hasMap ? PBR_MAP_TINT_FREE_HEX : SKIN_MATERIAL.colorHex;
}

export type StudioConfigError =
  | { readonly kind: "InvalidBackground" }
  | { readonly kind: "InvalidLightCount"; readonly received: number }
  | { readonly kind: "InvalidIntensity"; readonly light: StudioLightKind }
  | { readonly kind: "InvalidPosition"; readonly light: StudioLightKind }
  | { readonly kind: "InvalidProbeCount"; readonly received: number }
  | { readonly kind: "InvalidProbeMetal"; readonly probe: LightProbeName }
  | { readonly kind: "InvalidProbeShape"; readonly probe: LightProbeName };

function assertNever(value: never, message = "Unhandled case"): never {
  throw new Error(`${message}: ${JSON.stringify(value)}`);
}

/** Parse-once del fondo: solo el blanco estudio es valido. */
export function parseStudioBackground(raw: unknown): Result<StudioBackground, StudioConfigError> {
  if (typeof raw !== "number" || !Number.isInteger(raw)) {
    return { ok: false, error: { kind: "InvalidBackground" } };
  }
  if (raw !== STUDIO_BACKGROUND_HEX) {
    return { ok: false, error: { kind: "InvalidBackground" } };
  }
  return { ok: true, value: raw as StudioBackground };
}

/** El trio canonico, sin alocacion por llamada: la referencia const. */
export function buildStudioLights(): readonly StudioLight[] {
  return STUDIO_LIGHTS;
}

/** Chequeo total del trio: conteo exacto e intensidades sanas. */
export function checkStudioLights(
  lights: readonly StudioLight[],
): Result<readonly StudioLight[], StudioConfigError> {
  if (lights.length !== 3) {
    return { ok: false, error: { kind: "InvalidLightCount", received: lights.length } };
  }
  const kinds: readonly string[] = lights.map((l) => l.kind);
  if (kinds[0] !== "key" || kinds[1] !== "fill" || kinds[2] !== "rim") {
    return { ok: false, error: { kind: "InvalidLightCount", received: lights.length } };
  }
  for (const light of lights) {
    if (!Number.isFinite(light.intensity) || light.intensity <= 0 || light.intensity > 3) {
      return { ok: false, error: { kind: "InvalidIntensity", light: light.kind } };
    }
    for (const axis of light.position) {
      if (!Number.isFinite(axis)) {
        return { ok: false, error: { kind: "InvalidPosition", light: light.kind } };
      }
    }
  }
  return { ok: true, value: lights };
}

export type ViewerFaceStatus = "idle" | "loading" | "ready" | "invalid";

export type ViewerStatusError = { readonly kind: "UnknownStatus" };

/** Parse-once del estado del visor: vocabulario cerrado, sin `as`. */
export function parseViewerStatus(raw: unknown): Result<ViewerFaceStatus, ViewerStatusError> {
  if (typeof raw !== "string") return { ok: false, error: { kind: "UnknownStatus" } };
  switch (raw) {
    case "idle":
    case "loading":
    case "ready":
    case "invalid":
      return { ok: true, value: raw };
    default:
      return { ok: false, error: { kind: "UnknownStatus" } };
  }
}

/** Mensaje humano por estado; exhaustivo via assertNever. */
export function viewerStatusMessage(status: ViewerFaceStatus): string {
  switch (status) {
    case "idle":
      return "visor 3D listo (neutro, sin resultado)";
    case "loading":
      return "cargando cara real (GLB)...";
    case "ready":
      return "cara real lista: meshes A+B cargados en visor (GLB parseado + PBR).";
    case "invalid":
      return "GLB invalido: no se pudo parsear (revisa el zip).";
    default:
      return assertNever(status);
  }
}

/** Chequeo total del duo de sondas: nombres, metal y forma sanos. */
export function checkLightProbes(
  probes: readonly LightProbe[],
): Result<readonly LightProbe[], StudioConfigError> {
  if (probes.length !== 2) {
    return { ok: false, error: { kind: "InvalidProbeCount", received: probes.length } };
  }
  if (probes[0]?.name !== "chrome" || probes[1]?.name !== "graphite") {
    return { ok: false, error: { kind: "InvalidProbeCount", received: probes.length } };
  }
  for (const probe of probes) {
    if (!Number.isFinite(probe.metalness) || probe.metalness <= 0 || probe.metalness > 1) {
      return { ok: false, error: { kind: "InvalidProbeMetal", probe: probe.name } };
    }
    if (
      !Number.isFinite(probe.roughness) ||
      probe.roughness <= 0 ||
      probe.roughness >= 1 ||
      !Number.isFinite(probe.radius) ||
      probe.radius <= 0 ||
      probe.position.length !== 3
    ) {
      return { ok: false, error: { kind: "InvalidProbeShape", probe: probe.name } };
    }
    for (const axis of probe.position) {
      if (!Number.isFinite(axis)) {
        return { ok: false, error: { kind: "InvalidProbeShape", probe: probe.name } };
      }
    }
  }
  return { ok: true, value: probes };
}

export function studioConfigMessage(error: StudioConfigError): string {
  switch (error.kind) {
    case "InvalidBackground":
      return "fondo de estudio invalido: solo blanco 0xffffff";
    case "InvalidLightCount":
      return `estudio incompleto: se esperaban 3 luces, llegaron ${error.received}`;
    case "InvalidIntensity":
      return `luz ${error.light} con intensidad fuera de rango`;
    case "InvalidPosition":
      return `luz ${error.light} con posicion invalida (eje no finito)`;
    case "InvalidProbeCount":
      return `sondas incompletas: se esperaban cromo + grafito, llegaron ${error.received}`;
    case "InvalidProbeMetal":
      return `sonda ${error.probe} sin metal (metalness fuera de (0,1])`;
    case "InvalidProbeShape":
      return `sonda ${error.probe} con forma invalida (rugosidad/radio/posicion)`;
    default:
      return assertNever(error);
  }
}
