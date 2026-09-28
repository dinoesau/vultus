import JSZip from "jszip";

// Por qué jszip y no un parse manual del zip en el navegador: el resultado
// es un zip real (deflate + directorio central) generado por el worker.
// Un parser a mano es frágil ante cualquier cambio de compresión;
// jszip es el coste mínimo (~100KB) por robustez en Fase 1.

// Sin Magic Strings para rutas: un solo objeto con los paths del Seam HTTP.
export const API_PATHS = {
  compare: "/v1/compare",
  job: (id: string) => `/v1/jobs/${id}`,
  events: (id: string) => `/v1/jobs/${id}/events`,
  result: (id: string) => `/v1/jobs/${id}/result`,
} as const;

// Nombres exactos del bundle (contrato con el worker, zip-6 v2 sin heatmap:
// 2 UV + 2 GLB + 2 PBR en orden canonico python).
// Espejo de edge/contract.ts ZIP_MANIFEST; no renombrar sin cambiar el worker.
export const RESULT_FILES = {
  uvA: "uv_a.png",
  uvB: "uv_b.png",
  meshA: "mesh_a.glb",
  meshB: "mesh_b.glb",
  pbrA: "pbr_a.png",
  pbrB: "pbr_b.png",
} as const;

export type Result<T, E> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

export type ZipMissing = { readonly kind: "MissingEntry"; readonly name: string };

/** Parse-once del zip: entry ausente es Result, no throw. */
export async function tryPickEntry(
  zip: import("jszip"),
  name: string,
): Promise<Result<Blob, ZipMissing>> {
  const entry = zip.file(name);
  if (!entry) return { ok: false, error: { kind: "MissingEntry", name } };
  return { ok: true, value: await entry.async("blob") };
}

export interface JobEvent {
  job_id: string;
  status: string;
  progress: number;
  stage: string;
}

function isRecord(raw: unknown): raw is Record<string, unknown> {
  return typeof raw === "object" && raw !== null && !Array.isArray(raw);
}

// Fuente de verdad del ciclo en edge/contract.ts STATUSES; espejo local para
// no acoplar el bundle del navegador al worker. Regla unica aqui dentro.
const JOB_STATUSES = ["queued", "processing", "done", "failed", "expired"] as const;
export type JobStatus = (typeof JOB_STATUSES)[number];

export type JobEventError = { readonly kind: "InvalidJobEvent"; readonly detail: string };

/** Parse-once del evento WS: JSON crudo a JobEvent probado, sin `as`. */
export function parseJobEvent(raw: unknown): Result<JobEvent, JobEventError> {
  if (!isRecord(raw)) {
    return { ok: false, error: { kind: "InvalidJobEvent", detail: "event is not an object" } };
  }
  const jobId = raw["job_id"];
  if (typeof jobId !== "string" || jobId.length === 0) {
    return { ok: false, error: { kind: "InvalidJobEvent", detail: "missing job_id" } };
  }
  const status = raw["status"];
  if (typeof status !== "string" || !(JOB_STATUSES as readonly string[]).includes(status)) {
    return { ok: false, error: { kind: "InvalidJobEvent", detail: "invalid status" } };
  }
  const progress = raw["progress"];
  if (typeof progress !== "number" || !Number.isFinite(progress) || progress < 0 || progress > 1) {
    return { ok: false, error: { kind: "InvalidJobEvent", detail: "invalid progress" } };
  }
  const stage = raw["stage"];
  if (typeof stage !== "string" || stage.length === 0) {
    return { ok: false, error: { kind: "InvalidJobEvent", detail: "invalid stage" } };
  }
  return { ok: true, value: { job_id: jobId, status, progress, stage } };
}

export type CompareResponseError = { readonly kind: "InvalidCompareResponse" };

/** Parse-once del POST /v1/compare: exige job_id no vacio, sin `as`. */
export function parseCompareResponse(raw: unknown): Result<{ jobId: string }, CompareResponseError> {
  if (!isRecord(raw)) return { ok: false, error: { kind: "InvalidCompareResponse" } };
  const jobId = raw["job_id"];
  if (typeof jobId !== "string" || jobId.length === 0) {
    return { ok: false, error: { kind: "InvalidCompareResponse" } };
  }
  return { ok: true, value: { jobId } };
}

/** Detalle humano de un error del gateway sin filtrar internos. */
export function errorDetail(raw: unknown): string {
  if (isRecord(raw) && typeof raw["detail"] === "string") return raw["detail"];
  try {
    return JSON.stringify(raw);
  } catch {
    return "invalid response";
  }
}

export type PollStatusError = { readonly kind: "InvalidPollStatus" };

/** Parse-once del GET polling /v1/jobs/{id}: solo el status terminal importa. */
export function parsePollStatus(raw: unknown): Result<JobStatus, PollStatusError> {
  if (!isRecord(raw)) return { ok: false, error: { kind: "InvalidPollStatus" } };
  const status = raw["status"];
  if (typeof status !== "string" || !(JOB_STATUSES as readonly string[]).includes(status)) {
    return { ok: false, error: { kind: "InvalidPollStatus" } };
  }
  switch (status) {
    case "queued":
    case "processing":
    case "done":
    case "failed":
    case "expired":
      return { ok: true, value: status };
    default:
      return { ok: false, error: { kind: "InvalidPollStatus" } };
  }
}

const TERMINAL = new Set(["done", "failed", "expired"]);

export function isTerminal(status: string): boolean {
  return TERMINAL.has(status);
}

// Resultado aun no listo: 409 (not done) en prod vivo, 404 transitorio si el
// zip tarda en aparecer en R2 tras done. Ambos reintentan acotado, sin colgar la UI.
export function isRetryableResultStatus(status: number): boolean {
  return status === 404 || status === 409;
}

// Deriva la URL WS desde la HTTP cambiando el esquema (http->ws, https->wss).
export function toWsUrl(apiUrl: string, jobId: string): string {
  const base = apiUrl.replace(/^http:/, "ws:").replace(/^https:/, "wss:");
  return `${base}${API_PATHS.events(jobId)}`;
}

export function statusMessage(
  status: string,
  stage: string,
  progress: number,
  jobId: string,
): string {
  const pct = Math.round(progress * 100);
  switch (status) {
    case "queued":
      return `en cola (etapa ${stage}, ${pct}%)...`;
    case "processing":
      return `procesando: ${stage} ${pct}%`;
    case "done":
      return `listo (${stage}, 100%). Descargando resultado...`;
    case "failed":
      return `falló en etapa ${stage}. Revisa las imágenes e inténtalo de nuevo.`;
    case "expired":
      return `expiró (TTL 60s en servidor). Si ya descargaste el zip, sigue disponible abajo desde memoria.`;
    default:
      return `${status} ${stage} ${pct}%`;
  }
}

export interface ResultImages {
  uvA: Blob;
  uvB: Blob;
}

export interface ResultPbr {
  pbrA: Blob;
  pbrB: Blob;
}

export interface ResultMeshes {
  meshA: Blob;
  meshB: Blob;
}

// Version esperada del contrato (espejo de edge/contract.ts CONTRACT_VERSION;
// v2 = zip-6 sin heatmap). El frontend la valida contra GET /health.
// Bundle-split: edge y frontend son bundles separados, asi que el brand se
// espeja aqui (single mint via mintContractVersionUnchecked) en vez de importar edge.
// Fuente unica local: CONTRACT_VERSION; EXPECTED_CONTRACT_VERSION es alias compat.
type Brand<T, Name extends string> = T & { readonly __brand: Name };
declare const ContractVersionBrand: unique symbol;
export type ContractVersion = Brand<number, "ContractVersion"> & {
  readonly [ContractVersionBrand]: "ContractVersion";
};
export type ContractVersionError = { readonly kind: "InvalidContractVersion" };

// Single mint site en este bundle (espejo de edge/contract.ts): la asercion
// vive solo aqui, se revisa como sudo. Todo path smart-constructor la usa.
function mintContractVersionUnchecked(value: number): ContractVersion {
  return value as ContractVersion;
}

export const CONTRACT_VERSION: ContractVersion = mintContractVersionUnchecked(2);

export function contractVersionToNumber(v: ContractVersion): number {
  return v;
}

/** Fuente unica: parseContractVersion estricto entero >=1 via Result; nunca lanza. */
export function parseContractVersion(raw: unknown): Result<ContractVersion, ContractVersionError> {
  if (typeof raw !== "number" || !Number.isInteger(raw) || raw < 1) {
    return { ok: false, error: { kind: "InvalidContractVersion" } };
  }
  return { ok: true, value: mintContractVersionUnchecked(raw) };
}

// Alias compat: mismo brand que CONTRACT_VERSION, no segunda fuente.
export const EXPECTED_CONTRACT_VERSION: ContractVersion = CONTRACT_VERSION;

export type HealthVersionError =
  | { readonly kind: "InvalidHealthVersion" }
  | { readonly kind: "VersionMismatch"; readonly expected: ContractVersion; readonly received: ContractVersion };

/** Parse-once del GET /health: exige contract_version brand que iguale al esperado, sin `as`. */
export function parseHealthContractVersion(
  raw: unknown,
): Result<ContractVersion, HealthVersionError> {
  if (!isRecord(raw)) return { ok: false, error: { kind: "InvalidHealthVersion" } };
  const parsed = parseContractVersion(raw["contract_version"]);
  if (!parsed.ok) {
    return { ok: false, error: { kind: "InvalidHealthVersion" } };
  }
  if (contractVersionToNumber(parsed.value) !== contractVersionToNumber(CONTRACT_VERSION)) {
    return {
      ok: false,
      error: { kind: "VersionMismatch", expected: CONTRACT_VERSION, received: parsed.value },
    };
  }
  return { ok: true, value: parsed.value };
}

/** Cero panel roto: con mismatch o salud ilegible el visor muestra actualiza. */
export function healthNeedsUpdate(health: unknown): boolean {
  return !parseHealthContractVersion(health).ok;
}

export function contractVersionMessage(error: HealthVersionError): string {
  switch (error.kind) {
    case "InvalidHealthVersion":
      return "versión de contrato desconocida, actualiza el visor";
    case "VersionMismatch":
      return `contrato v${contractVersionToNumber(error.received)} distinto de v${contractVersionToNumber(error.expected)}, actualiza el visor`;
    default:
      return assertNever(error);
  }
}

// Unico sitio con `throw` por bug imposible en este bundle (espejo de
// edge/assert.ts; sin nuevas deps: edge y frontend son bundles separados).
// El dominio retorna Result, nunca lanza.
function assertNever(value: never, message = "Unhandled case"): never {
  throw new Error(`${message}: ${JSON.stringify(value)}`);
}

// Desempaqueta el zip en memoria; falla con mensaje claro si falta un PNG o GLB.
// Borde delgado: try* retorna Result. Pbr/Meshes no tienen wrapper throw
// (ZipMissing esperado va por Err); Images conserva su wrapper legacy solo por
// compat, el codigo nuevo debe usar tryExtract*. El core nuevo debe usar tryExtract*.
export async function tryExtractResultImages(zipBlob: Blob): Promise<Result<ResultImages, ZipMissing>> {
  const zip = await JSZip.loadAsync(zipBlob);
  const uvA = await tryPickEntry(zip, RESULT_FILES.uvA);
  if (!uvA.ok) return uvA;
  const uvB = await tryPickEntry(zip, RESULT_FILES.uvB);
  if (!uvB.ok) return uvB;
  return { ok: true, value: { uvA: uvA.value, uvB: uvB.value } };
}

export async function extractResultImages(zipBlob: Blob): Promise<ResultImages> {
  const parsed = await tryExtractResultImages(zipBlob);
  if (!parsed.ok) throw new Error(`el zip no contiene ${parsed.error.name}`);
  return parsed.value;
}

// Desempaqueta las texturas PBR en memoria; falla si falta un PBR.
// Railway total: solo try* retorna Result. Sin wrappers throw (ZipMissing es
// esperado y se propaga como Err al caller, nunca como excepcion).
export async function tryExtractResultPbr(zipBlob: Blob): Promise<Result<ResultPbr, ZipMissing>> {
  const zip = await JSZip.loadAsync(zipBlob);
  const pbrA = await tryPickEntry(zip, RESULT_FILES.pbrA);
  if (!pbrA.ok) return pbrA;
  const pbrB = await tryPickEntry(zip, RESULT_FILES.pbrB);
  if (!pbrB.ok) return pbrB;
  return { ok: true, value: { pbrA: pbrA.value, pbrB: pbrB.value } };
}

// Desempaqueta los meshes GLB en memoria; falla si falta un GLB.
// Railway total: solo try* retorna Result. Sin wrappers throw.
export async function tryExtractResultMeshes(zipBlob: Blob): Promise<Result<ResultMeshes, ZipMissing>> {
  const zip = await JSZip.loadAsync(zipBlob);
  const meshA = await tryPickEntry(zip, RESULT_FILES.meshA);
  if (!meshA.ok) return meshA;
  const meshB = await tryPickEntry(zip, RESULT_FILES.meshB);
  if (!meshB.ok) return meshB;
  return { ok: true, value: { meshA: meshA.value, meshB: meshB.value } };
}

// Parte visor (Wave 5): el zip-6 completo en una sola carga para pintar
// UV + PBR + meshes sin reparsear el zip tres veces. Result, no throw.
export interface ViewerBlobs {
  uvA: Blob;
  uvB: Blob;
  pbrA: Blob;
  pbrB: Blob;
  meshA: Blob;
  meshB: Blob;
}

export async function tryExtractViewerBlobs(
  zipBlob: Blob,
): Promise<Result<ViewerBlobs, ZipMissing>> {
  const zip = await JSZip.loadAsync(zipBlob);
  const uvA = await tryPickEntry(zip, RESULT_FILES.uvA);
  if (!uvA.ok) return uvA;
  const uvB = await tryPickEntry(zip, RESULT_FILES.uvB);
  if (!uvB.ok) return uvB;
  const pbrA = await tryPickEntry(zip, RESULT_FILES.pbrA);
  if (!pbrA.ok) return pbrA;
  const pbrB = await tryPickEntry(zip, RESULT_FILES.pbrB);
  if (!pbrB.ok) return pbrB;
  const meshA = await tryPickEntry(zip, RESULT_FILES.meshA);
  if (!meshA.ok) return meshA;
  const meshB = await tryPickEntry(zip, RESULT_FILES.meshB);
  if (!meshB.ok) return meshB;
  return {
    ok: true,
    value: {
      uvA: uvA.value,
      uvB: uvB.value,
      pbrA: pbrA.value,
      pbrB: pbrB.value,
      meshA: meshA.value,
      meshB: meshB.value,
    },
  };
}
