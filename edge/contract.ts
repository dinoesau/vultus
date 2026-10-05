/**
 * Contrato compartido edge (prod vivo). Fuente de verdad del contrato HTTP.
 * Python la importa como espejo, nunca al reves.
 * Constantes canonicas: MAX_IMAGE_BYTES, magic JPEG/PNG,
 * TtlSecs 1..=3600 default 60, Stage, JobId uuid.
 * Zip canonico v2: 6 piezas sin heatmap (ADR-008 revoca ADR-002).
 * Pre-validacion fina del gateway para no encolar basura a Queues+R2
 * y no diverger en mensajes 400.
 */
import { assertNever } from "./assert";

// Canon good-typescript: Brand generico zero-runtime. Los brands existentes
// (JobId/TtlSecs/Progress/ContractVersion) lo usan como alias canonico; se
// conserva el unique-symbol historico como interseccion para no romper
// asignabilidad. Declarado arriba del primer uso, single mint por tipo.
export type Brand<T, Name extends string> = T & { readonly __brand: Name };

export const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
export const RESULT_TTL_SECONDS = 60;
export const TTL_MIN_SECS = 1;
export const TTL_MAX_SECS = 3600;

export const STAGES = [
  "queued",
  "fit",
  "texture",
  "assemble",
  "done",
] as const;
export type StageName = (typeof STAGES)[number];

export const TERMINAL_STATUSES = ["done", "failed", "expired"] as const;
export type TerminalStatus = (typeof TERMINAL_STATUSES)[number];

export const STATUSES = ["queued", "processing", "done", "failed", "expired"] as const;
export type JobStatus = (typeof STATUSES)[number];
export type JobStatusError = { readonly kind: "InvalidStatus" };

// Hitos de progreso del pipeline GNM (espejo de backend/domain.py).
// Fuente TS unica: el pipeline Python los importa como literales del contrato.
export const PROGRESS_FIT = 0.4;
export const PROGRESS_TEXTURE = 0.75;
export const PROGRESS_ASSEMBLE = 0.95;
export const PROGRESS_DONE = 1.0;

// Islas UV GNM publicas (1-5). Mas alla es investigacion fuera de alcance.
export const GNM_ISLANDS = [1, 2, 3, 4, 5] as const;
export type GnmIsland = (typeof GNM_ISLANDS)[number];

// Manifiesto zip versionado: fuente unica que Python espeja.
// Zip-6 v2 sin heatmap (ADR-008): uv + mesh + PBR, en orden canonico python.
export const ZIP_MANIFEST = {
  uvA: "uv_a.png",
  uvB: "uv_b.png",
  meshA: "mesh_a.glb",
  meshB: "mesh_b.glb",
  pbrA: "pbr_a.png",
  pbrB: "pbr_b.png",
} as const;

export const ZIP_NAMES = [
  ZIP_MANIFEST.uvA,
  ZIP_MANIFEST.uvB,
  ZIP_MANIFEST.meshA,
  ZIP_MANIFEST.meshB,
  ZIP_MANIFEST.pbrA,
  ZIP_MANIFEST.pbrB,
] as const;
export type ZipName = (typeof ZIP_NAMES)[number];

// Track v3 FFHQ-UV figure (paralelo, sin tocar la via forense v2).
// Revoca solo para v3: ADR-009 (FLAME 5023), TTL 60 y zip-6.
// Migracion versionada: V3_CONTRACT_VERSION = 3 (forense sigue en 2).
// Bundle v3 de 6 piezas (malla densa + albedo 1024 + neutral + 3 relights
// con esferas como la figura). Nombres disjuntos del zip-6: prohibido mezclar.
export const V3_CONTRACT_VERSION_NUMBER = 3;
export const V3_VERT_COUNT = 20481;
export const V3_TRI_COUNT = 40832;
export const V3_UV_SIZE = 1024;
export const V3_TV_MAX = 2.0;
export const V3_RETENTION_DAYS = 7;

export const V3_ZIP_MANIFEST = {
  meshDense: "mesh_dense.glb",
  albedo: "albedo_1024.png",
  neutral: "relight_neutral.png",
  key: "relight_key.png",
  fill: "relight_fill.png",
  rim: "relight_rim.png",
} as const;

export const V3_ZIP_NAMES = [
  V3_ZIP_MANIFEST.meshDense,
  V3_ZIP_MANIFEST.albedo,
  V3_ZIP_MANIFEST.neutral,
  V3_ZIP_MANIFEST.key,
  V3_ZIP_MANIFEST.fill,
  V3_ZIP_MANIFEST.rim,
] as const;
export type V3ZipName = (typeof V3_ZIP_NAMES)[number];

// Version de contrato: v1 legacy era 7 piezas con heatmap sin versionar;
// v2 es zip-6 sin heatmap. El transporte (sobre) no cambia, el payload si.
declare const ContractVersionBrand: unique symbol;
export type ContractVersion = Brand<number, "ContractVersion"> & {
  readonly [ContractVersionBrand]: "ContractVersion";
};
export type ContractVersionError = { readonly kind: "InvalidContractVersion" };

export type ContractVersionMismatch =
  | { readonly kind: "Match" }
  | {
      readonly kind: "VersionMismatch";
      readonly expected: ContractVersion;
      readonly received: ContractVersion;
    };

// Single mint site per good-typescript pillar 1: the brand assertion lives only here,
// reviewed as sudo. All contract-version smart-constructor paths mint via this helper.
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

/** ADT total: Match solo si igual a CONTRACT_VERSION, si no VersionMismatch. */
export function checkContractVersion(received: ContractVersion): ContractVersionMismatch {
  if (contractVersionToNumber(received) === contractVersionToNumber(CONTRACT_VERSION)) {
    return { kind: "Match" };
  }
  return { kind: "VersionMismatch", expected: CONTRACT_VERSION, received };
}

export function contractMismatchToMessage(mismatch: ContractVersionMismatch): string {
  switch (mismatch.kind) {
    case "Match":
      return "contract version match";
    case "VersionMismatch":
      return `contract version mismatch: expected ${contractVersionToNumber(mismatch.expected)} got ${contractVersionToNumber(mismatch.received)}, actualiza el visor`;
    default:
      return assertNever(mismatch);
  }
}

export type Result<T, E> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

declare const JobIdBrand: unique symbol;
export type JobId = Brand<string, "JobId"> & { readonly [JobIdBrand]: "JobId" };
export type JobIdError = { readonly kind: "InvalidJobId" };

declare const TtlSecsBrand: unique symbol;
export type TtlSecs = Brand<number, "TtlSecs"> & { readonly [TtlSecsBrand]: "TtlSecs" };

declare const ProgressBrand: unique symbol;
export type Progress = Brand<number, "Progress"> & { readonly [ProgressBrand]: "Progress" };
export type ProgressError = { readonly kind: "InvalidProgress" };

export type StageError = { readonly kind: "InvalidStage" };

export function isTerminalStatus(s: unknown): s is TerminalStatus {
  return typeof s === "string" && (TERMINAL_STATUSES as readonly string[]).includes(s);
}

export function isJobStatus(s: unknown): s is JobStatus {
  return parseJobStatus(s).ok;
}

export function parseJobStatus(raw: unknown): Result<JobStatus, JobStatusError> {
  if (typeof raw !== "string" || !(STATUSES as readonly string[]).includes(raw)) {
    return { ok: false, error: { kind: "InvalidStatus" } };
  }
  return { ok: true, value: raw as JobStatus };
}

export function isUuid(s: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(s);
}

export function parseJobId(raw: unknown): Result<JobId, JobIdError> {
  if (typeof raw !== "string") return { ok: false, error: { kind: "InvalidJobId" } };
  const trimmed = raw.trim();
  if (!isUuid(trimmed)) return { ok: false, error: { kind: "InvalidJobId" } };
  return { ok: true, value: trimmed as JobId };
}

export function jobIdToString(id: JobId): string {
  return id;
}

export function isJpeg(bytes: Uint8Array): boolean {
  return bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
}

export function isPng(bytes: Uint8Array): boolean {
  return (
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a
  );
}

/** Paridad con `ImageBytesRef::parse`: size + magic, sin heap extra. */
export function hasSupportedMagic(bytes: Uint8Array): boolean {
  return isJpeg(bytes) || isPng(bytes);
}

export type TtlError = { readonly kind: "InvalidTtlSecs" };

// Single mint site per good-typescript pillar 1: the brand assertion lives only here,
// reviewed as sudo. All TTL smart-constructor paths mint via this helper.
function mintTtlSecsUnchecked(value: number): TtlSecs {
  return value as TtlSecs;
}

/** Fuente unica: parseTtlSecs estricto 1..=3600 via Result InvalidTtlSecs; shell default 60 con log, DO 400. Nunca lanza. */
export function parseTtlSecs(raw: unknown): Result<TtlSecs, TtlError> {
  const err: Result<TtlSecs, TtlError> = { ok: false, error: { kind: "InvalidTtlSecs" } };
  if (typeof raw === "number") {
    if (!Number.isInteger(raw)) return err;
    if (raw < TTL_MIN_SECS || raw > TTL_MAX_SECS) return err;
    return { ok: true, value: mintTtlSecsUnchecked(raw) };
  }
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!/^(0|[1-9][0-9]*)$/.test(trimmed)) return err;
    const n = Number(trimmed);
    if (!Number.isInteger(n) || n < TTL_MIN_SECS || n > TTL_MAX_SECS) return err;
    return { ok: true, value: mintTtlSecsUnchecked(n) };
  }
  return err;
}

export function resolveTtlSecs(raw: unknown): { ttl: TtlSecs; invalid: boolean; raw: unknown } {
  const r = parseTtlSecs(raw);
  if (r.ok) return { ttl: r.value, invalid: false, raw };
  return { ttl: mintTtlSecsUnchecked(RESULT_TTL_SECONDS), invalid: true, raw };
}

export function ttlToNumber(ttl: TtlSecs): number {
  return ttl;
}

/** Fuente unica: parseProgress. isValid* delega aqui para no duplicar 0..1. */
export function isValidProgress(n: unknown): n is number {
  return parseProgress(n).ok;
}

export function parseProgress(raw: unknown): Result<Progress, ProgressError> {
  if (typeof raw !== "number" || !Number.isFinite(raw) || raw < 0 || raw > 1) {
    return { ok: false, error: { kind: "InvalidProgress" } };
  }
  return { ok: true, value: raw as Progress };
}

export function progressToNumber(p: Progress): number {
  return p;
}

/** Fuente unica: parseStage. isValidStage delega aqui para no duplicar STAGES. */
export function isValidStage(s: unknown): s is StageName {
  return parseStage(s).ok;
}

export function parseStage(raw: unknown): Result<StageName, StageError> {
  if (typeof raw !== "string" || !(STAGES as readonly string[]).includes(raw)) {
    return { ok: false, error: { kind: "InvalidStage" } };
  }
  return { ok: true, value: raw as StageName };
}
