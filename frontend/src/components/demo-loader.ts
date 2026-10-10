import type { Result } from "./compare-client.ts";

// Por qué módulo propio y no fetch inline en index.astro: las rutas demo,
// el parse y los nombres de archivo viven en un solo sitio testeable sin
// DOM. index.astro solo cablea inputs + previews + botón (ver plan-122).

// Sin Magic Strings para rutas: un solo objeto con los paths del seam demo.
// Espejo de `frontend/public/demo/README.md`; no renombrar sin mover assets.
export const DEMO_PATHS = {
  a: "/demo/obama-a.jpg",
  b: "/demo/obama-b.jpg",
} as const;

export type DemoPaths = { readonly a: string; readonly b: string };

// Nombres fijos para el FormData: el meta del slot muestra de dónde vino
// cada archivo y el gateway recibe filename con sentido.
export const DEMO_FILE_NAMES = {
  a: "demo-obama-a.jpg",
  b: "demo-obama-b.jpg",
} as const;

// Magics espejo de backend/domain.py JPEG_MAGIC/PNG_MAGIC: el gateway
// revalida de todos modos; aquí solo se filtra basura temprano.
const JPEG_MAGIC = [0xff, 0xd8, 0xff] as const;
const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] as const;

// Bundle-split: edge y frontend son bundles separados, así que el brand se
// espeja aquí (single mint via mintDemoImageUnchecked), igual que
// CONTRACT_VERSION en compare-client.ts.
type Brand<T, Name extends string> = T & { readonly __brand: Name };
declare const DemoImageBrand: unique symbol;
export type DemoImage = Brand<Blob, "DemoImage"> & {
  readonly [DemoImageBrand]: "DemoImage";
};

// Single mint site en este módulo: la aserción vive solo aquí,
// se revisa como sudo. Todo path smart-constructor la usa.
function mintDemoImageUnchecked(value: Blob): DemoImage {
  return value as DemoImage;
}

export function demoImageToBlob(image: DemoImage): Blob {
  return image;
}

export type DemoImageError = {
  readonly kind: "InvalidDemoBlob";
  readonly reason: "empty" | "type" | "magic";
};

function hasPrefix(bytes: Uint8Array, magic: readonly number[]): boolean {
  if (bytes.length < magic.length) return false;
  return magic.every((byte, i) => bytes[i] === byte);
}

/** Parse-once del blob demo a DemoImage probada. Total, nunca lanza. */
export async function parseDemoBlob(raw: unknown): Promise<Result<DemoImage, DemoImageError>> {
  if (!(raw instanceof Blob)) {
    return { ok: false, error: { kind: "InvalidDemoBlob", reason: "type" } };
  }
  if (raw.size === 0) {
    return { ok: false, error: { kind: "InvalidDemoBlob", reason: "empty" } };
  }
  const isJpeg = raw.type === "image/jpeg";
  const isPng = raw.type === "image/png";
  if (!isJpeg && !isPng) {
    return { ok: false, error: { kind: "InvalidDemoBlob", reason: "type" } };
  }
  let head: Uint8Array;
  try {
    head = new Uint8Array(await raw.slice(0, 8).arrayBuffer());
  } catch {
    return { ok: false, error: { kind: "InvalidDemoBlob", reason: "magic" } };
  }
  const magicOk = isJpeg ? hasPrefix(head, JPEG_MAGIC) : hasPrefix(head, PNG_MAGIC);
  if (!magicOk) {
    return { ok: false, error: { kind: "InvalidDemoBlob", reason: "magic" } };
  }
  return { ok: true, value: mintDemoImageUnchecked(raw) };
}

/** Config por env con default estático (12-Factor III). Total, nunca lanza. */
export function resolveDemoPaths(env: unknown): DemoPaths {
  const get = (key: string): string | undefined => {
    if (typeof env !== "object" || env === null) return undefined;
    const value = (env as Record<string, unknown>)[key];
    return typeof value === "string" && value.length > 0 ? value : undefined;
  };
  return {
    a: get("VITE_DEMO_A") ?? DEMO_PATHS.a,
    b: get("VITE_DEMO_B") ?? DEMO_PATHS.b,
  };
}

export type DemoPair = { readonly a: DemoImage; readonly b: DemoImage };

export type DemoLoadError = {
  readonly kind: "DemoLoadFailed";
  readonly which: "a" | "b";
  readonly detail: "network" | "status" | "blob";
  readonly status?: number;
};

export interface DemoLoadOptions {
  readonly fetchImpl?: typeof fetch;
  readonly signal?: AbortSignal;
}

async function loadOne(
  which: "a" | "b",
  url: string,
  options: DemoLoadOptions,
): Promise<Result<DemoImage, DemoLoadError>> {
  const fetchImpl = options.fetchImpl ?? fetch;
  let res: Response;
  try {
    res = await fetchImpl(url, { signal: options.signal });
  } catch {
    return { ok: false, error: { kind: "DemoLoadFailed", which, detail: "network" } };
  }
  if (!res.ok) {
    return {
      ok: false,
      error: { kind: "DemoLoadFailed", which, detail: "status", status: res.status },
    };
  }
  let blob: Blob;
  try {
    blob = await res.blob();
  } catch {
    return { ok: false, error: { kind: "DemoLoadFailed", which, detail: "blob" } };
  }
  const parsed = await parseDemoBlob(blob);
  if (!parsed.ok) {
    return { ok: false, error: { kind: "DemoLoadFailed", which, detail: "blob" } };
  }
  return parsed;
}

/**
 * Carga el par demo en paralelo. Railway total: el par llega probado o
 * llega un único Err clasificado por slot. Nunca lanza.
 * El mensaje visible ante Err lo posee #123; aquí solo se clasifica.
 */
export async function loadDemoBlobs(
  paths: DemoPaths,
  options: DemoLoadOptions = {},
): Promise<Result<DemoPair, DemoLoadError>> {
  const [a, b] = await Promise.all([
    loadOne("a", paths.a, options),
    loadOne("b", paths.b, options),
  ]);
  if (!a.ok) return a;
  if (!b.ok) return b;
  return { ok: true, value: { a: a.value, b: b.value } };
}

export type DemoFiles = { readonly fileA: File; readonly fileB: File };

/** DemoImage probadas a Files listos para DataTransfer + FormData. Total. */
export function blobsToFiles(a: DemoImage, b: DemoImage): DemoFiles {
  const blobA = demoImageToBlob(a);
  const blobB = demoImageToBlob(b);
  return {
    fileA: new File([blobA], DEMO_FILE_NAMES.a, { type: blobA.type || "image/jpeg" }),
    fileB: new File([blobB], DEMO_FILE_NAMES.b, { type: blobB.type || "image/jpeg" }),
  };
}
