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
export type DemoSlot = "a" | "b";

export type DemoMime = "image/jpeg" | "image/png";

// Magics espejo de backend/domain.py JPEG_MAGIC/PNG_MAGIC. El MIME se
// olfatea por contenido, nunca por `Blob.type` (viene del Content-Type del
// host y miente: `""`, `image/jpg`, `application/octet-stream` son reales).
// El gateway revalida de todos modos; aquí solo se filtra basura temprano.
const JPEG_MAGIC = [0xff, 0xd8, 0xff] as const;
const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] as const;

// Brand local de un solo mint (ver mintDemoImageUnchecked). El `__brand`
// impide pasar un Blob crudo donde se exige DemoImage (ver test negativo
// con @ts-expect-error en demo-loader.test.ts).
type Brand<T, Name extends string> = T & { readonly __brand: Name };
export type DemoImage = Brand<Blob, "DemoImage">;

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

/**
 * Olfatea el MIME por magic, nunca por `Blob.type`. Total, nunca lanza.
 * `null` = bytes que no son JPEG ni PNG.
 */
export async function sniffDemoMime(raw: Blob): Promise<DemoMime | null> {
  let head: Uint8Array;
  try {
    head = new Uint8Array(await raw.slice(0, 8).arrayBuffer());
  } catch {
    return null;
  }
  if (hasPrefix(head, JPEG_MAGIC)) return "image/jpeg";
  if (hasPrefix(head, PNG_MAGIC)) return "image/png";
  return null;
}

/** Parse-once del blob demo a DemoImage probada. Total, nunca lanza. */
export async function parseDemoBlob(raw: unknown): Promise<Result<DemoImage, DemoImageError>> {
  if (!(raw instanceof Blob)) {
    return { ok: false, error: { kind: "InvalidDemoBlob", reason: "type" } };
  }
  if (raw.size === 0) {
    return { ok: false, error: { kind: "InvalidDemoBlob", reason: "empty" } };
  }
  const mime = await sniffDemoMime(raw);
  if (mime === null) {
    return { ok: false, error: { kind: "InvalidDemoBlob", reason: "magic" } };
  }
  return { ok: true, value: mintDemoImageUnchecked(raw) };
}

export type DemoPair = { readonly a: DemoImage; readonly b: DemoImage };

export type DemoLoadError = {
  readonly kind: "DemoLoadFailed";
  readonly which: DemoSlot;
  readonly detail: "network" | "status" | "blob-transport" | "blob-invalid";
  readonly status?: number;
};

// Fuente única del mensaje visible ante fallo demo (#123). El loader
// clasifica el Err; la presentación lo pinta sin ramificar por detail/slot.
export const DEMO_LOAD_ERROR_MESSAGE = "Could not load demo. Try again" as const;

export type DemoDomError = { readonly kind: "DemoDomError" };

export type DemoPresentableError = DemoLoadError | DemoImageError | DemoDomError;

/**
 * Mapper total del Err demo al mensaje visible. Un solo literal para los
 * 4 details + blob-invalid + DOM, así el e2e y la UI no duplican strings.
 * Nunca lanza.
 */
export function demoLoadErrorMessage(_err: DemoPresentableError): typeof DEMO_LOAD_ERROR_MESSAGE {
  void _err;
  return DEMO_LOAD_ERROR_MESSAGE;
}

export interface DemoLoadOptions {
  readonly fetchImpl?: typeof fetch;
  readonly signal?: AbortSignal;
}

async function loadOne(
  which: DemoSlot,
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
    return { ok: false, error: { kind: "DemoLoadFailed", which, detail: "blob-transport" } };
  }
  const parsed = await parseDemoBlob(blob);
  if (!parsed.ok) {
    return { ok: false, error: { kind: "DemoLoadFailed", which, detail: "blob-invalid" } };
  }
  return parsed;
}

/**
 * Carga el par demo en paralelo. Railway total: el par llega probado o
 * llega un único Err clasificado por slot. Prioridad documentada: ante
 * doble fallo gana el slot `a` (el primero que se pide). Nunca lanza.
 * El pintado del Err vive en `index.astro` vía `demoLoadErrorMessage`;
 * aquí solo se clasifica y se expone el literal único.
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

/** Nombre del archivo por slot y MIME real: extensión y bytes coinciden. */
export function demoFileName(which: DemoSlot, mime: DemoMime): string {
  const ext = mime === "image/png" ? "png" : "jpg";
  return `demo-obama-${which}.${ext}`;
}

/**
 * DemoImages probadas a Files listos para DataTransfer + FormData. Total:
 * el MIME se re-olfatea del contenido (el `Blob.type` no es confiable) y el
 * nombre acompaña al MIME. El `null` es inalcanzable para DemoImage (el
 * brand solo nace en parseDemoBlob con magic válido) pero se propaga como
 * Err en vez de fallback muerto o throw.
 */
export async function blobsToFiles(
  a: DemoImage,
  b: DemoImage,
): Promise<Result<DemoFiles, DemoImageError>> {
  const [mimeA, mimeB] = await Promise.all([
    sniffDemoMime(demoImageToBlob(a)),
    sniffDemoMime(demoImageToBlob(b)),
  ]);
  if (mimeA === null || mimeB === null) {
    return { ok: false, error: { kind: "InvalidDemoBlob", reason: "magic" } };
  }
  const blobA = demoImageToBlob(a);
  const blobB = demoImageToBlob(b);
  return {
    ok: true,
    value: {
      fileA: new File([blobA], demoFileName("a", mimeA), { type: mimeA }),
      fileB: new File([blobB], demoFileName("b", mimeB), { type: mimeB }),
    },
  };
}
