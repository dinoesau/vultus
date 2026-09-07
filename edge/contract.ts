/**
 * Contrato compartido edge (prod vivo). Fuente de verdad del contrato HTTP.
 * Python la importa como espejo, nunca al reves.
 * Constantes canonicas: MAX_IMAGE_BYTES, magic JPEG/PNG,
 * TtlSecs 1..=3600 default 60, Stage, JobId uuid.
 * Pre-validacion fina del gateway para no encolar basura a Queues+R2
 * y no diverger en mensajes 400.
 */

export const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
export const RESULT_TTL_SECONDS = 60;
export const TTL_MIN_SECS = 1;
export const TTL_MAX_SECS = 3600;

export const STAGES = [
  "queued",
  "landmarks",
  "flame",
  "freeuv",
  "bake",
  "done",
] as const;
export type StageName = (typeof STAGES)[number];

export const TERMINAL_STATUSES = ["done", "failed", "expired"] as const;
export type TerminalStatus = (typeof TERMINAL_STATUSES)[number];

export type Result<T, E> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

declare const JobIdBrand: unique symbol;
export type JobId = string & { readonly [JobIdBrand]: "JobId" };
export type JobIdError = { readonly kind: "InvalidJobId" };

declare const TtlSecsBrand: unique symbol;
export type TtlSecs = number & { readonly [TtlSecsBrand]: "TtlSecs" };

declare const ProgressBrand: unique symbol;
export type Progress = number & { readonly [ProgressBrand]: "Progress" };
export type ProgressError = { readonly kind: "InvalidProgress" };

export type StageError = { readonly kind: "InvalidStage" };

export function isTerminalStatus(s: unknown): s is TerminalStatus {
  return typeof s === "string" && (TERMINAL_STATUSES as readonly string[]).includes(s);
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

/** Paridad con `TtlSecs`: clamp 1..=3600, default 60. Nunca NaN. */
export function parseTtlSecs(raw: string | null | undefined): number {
  const n = Number(raw ?? `${RESULT_TTL_SECONDS}`);
  if (!Number.isFinite(n)) return RESULT_TTL_SECONDS;
  const floored = Math.floor(n);
  if (floored < TTL_MIN_SECS) return TTL_MIN_SECS;
  if (floored > TTL_MAX_SECS) return TTL_MAX_SECS;
  return floored;
}

export function parseTtlSecsBranded(raw: unknown): TtlSecs {
  const asString = typeof raw === "string" ? raw : raw == null ? null : String(raw);
  const clamped = parseTtlSecs(asString);
  return clamped as TtlSecs;
}

export function ttlToNumber(ttl: TtlSecs): number {
  return ttl;
}

export function isValidProgress(n: unknown): n is number {
  return typeof n === "number" && Number.isFinite(n) && n >= 0 && n <= 1;
}

export function parseProgress(raw: unknown): Result<Progress, ProgressError> {
  if (!isValidProgress(raw)) return { ok: false, error: { kind: "InvalidProgress" } };
  return { ok: true, value: raw as Progress };
}

export function progressToNumber(p: Progress): number {
  return p;
}

export function isValidStage(s: unknown): s is StageName {
  return typeof s === "string" && (STAGES as readonly string[]).includes(s);
}

export function parseStage(raw: unknown): Result<StageName, StageError> {
  if (!isValidStage(raw)) return { ok: false, error: { kind: "InvalidStage" } };
  return { ok: true, value: raw };
}
