/**
 * Gateway fino Cloudflare (prod vivo).
 * La API local vive en Python FastAPI para paridad dev; en prod el trafico
 * va edge -> Queues+R2 -> Modal workers -> R2 -> edge.
 * Pre-validacion del contrato (`edge/contract.ts`, fuente de verdad)
 * via `contract.ts` para no encolar basura a Queues+R2 y no diverger en mensajes 400.
 * En prod exige bindings reales (R2 + Queue + DO); sin fallbacks dummy.
 *
 * Shell delgado (parse-once): cada input se parsea una sola vez en el borde a
 * brands probados (`parseJobId`, `parseProgress`, `parseStage`,
 * `parseTtlSecsBranded`); aguas abajo solo viajan `JobId`, `Progress`,
 * `StageName` y `TtlSecs`, sin `isValid*` ni `as`. Errores estratificados en
 * `WorkerDomainError` (4xx por variante) vs `AppError` infra (500/502 con log).
 * Sin type-state: las rutas no forman un workflow ordenado con operaciones
 * distintas; el ciclo queued -> done vive en el DO.
 */
import {
  MAX_IMAGE_BYTES,
  hasSupportedMagic,
  jobIdToString,
  parseJobId,
  parseProgress,
  parseStage,
  parseTtlSecsBranded,
  progressToNumber,
  ttlToNumber,
  type JobId,
  type Progress,
  type Result,
  type StageName,
  type TtlSecs,
} from "./contract";

interface Env {
  VULTUS_QUEUE: Queue;
  VULTUS_BUCKET: R2Bucket;
  VULTUS_PROGRESS: DurableObjectNamespace;
  QUEUE_DRIVER?: string;
  R2_TTL_SECONDS?: string;
}

export { ProgressDO } from "./progress-do";

// CORS permisivo como la API local Python:
// el sitio estatico (Pages) y la API (Worker) viven en origenes distintos.
// el sitio estatico (Pages) y la API (Worker) viven en origenes distintos.
// Sin estos headers el navegador bloquea el 202 aunque el job se encole.
const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
  "Access-Control-Max-Age": "86400",
};

function json(data: unknown, status = 200): Response {
  return Response.json(data, { status, headers: { ...CORS_HEADERS } });
}

function assertNever(value: never, message = "Unhandled case"): never {
  throw new Error(`${message}: ${JSON.stringify(value)}`);
}

type ImageProblem = "size" | "magic";

type WorkerDomainError =
  | { readonly kind: "MissingMultipart" }
  | { readonly kind: "MissingImages" }
  | { readonly kind: "InvalidImage"; readonly reason: ImageProblem }
  | { readonly kind: "InvalidJobId" }
  | { readonly kind: "InvalidJson" }
  | { readonly kind: "InvalidProgress" }
  | { readonly kind: "InvalidStage" }
  | { readonly kind: "NotFound" }
  | { readonly kind: "NotDone" };

type AppError =
  | { readonly kind: "Domain"; readonly error: WorkerDomainError }
  | { readonly kind: "MissingBindings"; readonly cause: string }
  | { readonly kind: "CorruptProgress"; readonly cause: string };

function domainToStatus(error: WorkerDomainError): number {
  switch (error.kind) {
    case "MissingMultipart":
    case "MissingImages":
    case "InvalidImage":
    case "InvalidJobId":
    case "InvalidJson":
    case "InvalidProgress":
    case "InvalidStage":
      return 400;
    case "NotFound":
      return 404;
    case "NotDone":
      return 409;
    default:
      return assertNever(error);
  }
}

function domainToMessage(error: WorkerDomainError): string {
  switch (error.kind) {
    case "MissingMultipart":
      return "missing multipart";
    case "MissingImages":
      return "missing image_a or image_b";
    case "InvalidImage":
      return error.reason === "size"
        ? "invalid image: size out of range"
        : "invalid image: not jpeg nor png";
    case "InvalidJobId":
      return "invalid job_id";
    case "InvalidJson":
      return "invalid json";
    case "InvalidProgress":
      return "invalid progress";
    case "InvalidStage":
      return "invalid stage";
    case "NotFound":
      return "not found";
    case "NotDone":
      return "not done";
    default:
      return assertNever(error);
  }
}

function reportAppError(error: AppError): Response {
  if (error.kind === "Domain") {
    return json({ detail: domainToMessage(error.error) }, domainToStatus(error.error));
  }
  if (error.kind === "CorruptProgress") {
    console.error("corrupt progress payload", { cause: error.cause });
    return json({ detail: "upstream error" }, 502);
  }
  if (error.kind === "MissingBindings") {
    console.error("infrastructure failure", { kind: error.kind, cause: error.cause });
    return json({ detail: "missing bindings" }, 500);
  }
  return assertNever(error);
}

function domainResponse(error: WorkerDomainError): Response {
  return reportAppError({ kind: "Domain", error });
}

function isRecord(raw: unknown): raw is Record<string, unknown> {
  return typeof raw === "object" && raw !== null && !Array.isArray(raw);
}

/** Unico sitio que convierte un segmento de ruta en `JobId` probado. */
function routeJobId(raw: unknown): Result<JobId, WorkerDomainError> {
  const parsed = parseJobId(raw);
  if (!parsed.ok) return { ok: false, error: { kind: "InvalidJobId" } };
  return parsed;
}

function newJobId(): JobId {
  const parsed = parseJobId(crypto.randomUUID());
  // `crypto.randomUUID` siempre emite UUID v4; si falla es un bug, no input de usuario.
  if (!parsed.ok) throw new Error("crypto.randomUUID produced an invalid job_id");
  return parsed.value;
}

interface CompareBuffers {
  readonly aBuf: ArrayBuffer;
  readonly bBuf: ArrayBuffer;
}

/** Forma + tamano + magic en una sola lectura; los buffers viajan probados a R2. */
async function loadCompareBuffers(form: FormData): Promise<Result<CompareBuffers, WorkerDomainError>> {
  const a: unknown = form.get("image_a");
  const b: unknown = form.get("image_b");
  if (!(a instanceof File) || !(b instanceof File)) {
    return { ok: false, error: { kind: "MissingImages" } };
  }
  if (a.size === 0 || b.size === 0 || a.size > MAX_IMAGE_BYTES || b.size > MAX_IMAGE_BYTES) {
    return { ok: false, error: { kind: "InvalidImage", reason: "size" } };
  }
  // Paridad con `is_jpeg` / `is_png` en `contract.ts`: leer una vez y reusar para R2.
  const aBuf = await a.arrayBuffer();
  const bBuf = await b.arrayBuffer();
  if (!hasSupportedMagic(new Uint8Array(aBuf)) || !hasSupportedMagic(new Uint8Array(bBuf))) {
    return { ok: false, error: { kind: "InvalidImage", reason: "magic" } };
  }
  return { ok: true, value: { aBuf, bBuf } };
}

interface ProgressUpdate {
  readonly progress?: Progress;
  readonly stage?: StageName;
}

/** Valida progress/stage con los smart constructors; `status` lo duena el DO. */
function parseProgressUpdate(raw: unknown): Result<ProgressUpdate, WorkerDomainError> {
  if (!isRecord(raw)) return { ok: false, error: { kind: "InvalidJson" } };
  const progressRaw: unknown = raw["progress"];
  const stageRaw: unknown = raw["stage"];
  let progress: Progress | undefined;
  let stage: StageName | undefined;
  if (progressRaw !== undefined) {
    const parsed = parseProgress(progressRaw);
    if (!parsed.ok) return { ok: false, error: { kind: "InvalidProgress" } };
    progress = parsed.value;
  }
  if (stageRaw !== undefined) {
    const parsed = parseStage(stageRaw);
    if (!parsed.ok) return { ok: false, error: { kind: "InvalidStage" } };
    stage = parsed.value;
  }
  return { ok: true, value: { progress, stage } };
}

/**
 * Reenvio normalizado desde brands; `status` pasa opaco al DO que lo valida.
 * Frontera de proceso distinta: el DO revalida, no es reparseo del hot path.
 */
function progressUpdateBody(update: ProgressUpdate, raw: unknown): string {
  const statusRaw: unknown = isRecord(raw) ? raw["status"] : undefined;
  return JSON.stringify({
    ...(update.progress !== undefined ? { progress: progressToNumber(update.progress) } : {}),
    ...(update.stage !== undefined ? { stage: update.stage } : {}),
    ...(statusRaw !== undefined ? { status: statusRaw } : {}),
  });
}

interface DoStatus {
  readonly jobId: string;
  readonly status: string;
}

/** Estrecha el JSON del DO sin `as`: forma rota es infra corrupta, no 404. */
function parseDoStatus(raw: unknown): Result<DoStatus, AppError> {
  if (!isRecord(raw)) {
    return { ok: false, error: { kind: "CorruptProgress", cause: "do status is not a record" } };
  }
  const jobId: unknown = raw["job_id"];
  const status: unknown = raw["status"];
  if (typeof jobId !== "string" || typeof status !== "string") {
    return {
      ok: false,
      error: { kind: "CorruptProgress", cause: "do status missing job_id/status strings" },
    };
  }
  return { ok: true, value: { jobId, status } };
}

/** Lee el body del DO como total: JSON roto tambien es `CorruptProgress`. */
async function readDoStatus(res: Response): Promise<Result<DoStatus, AppError>> {
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return { ok: false, error: { kind: "CorruptProgress", cause: "do status is not valid json" } };
  }
  return parseDoStatus(body);
}

async function handleCompare(req: Request, env: Env): Promise<Response> {
  if (!env.VULTUS_BUCKET || !env.VULTUS_QUEUE || !env.VULTUS_PROGRESS) {
    return reportAppError({
      kind: "MissingBindings",
      cause: "compare requires VULTUS_BUCKET, VULTUS_QUEUE and VULTUS_PROGRESS",
    });
  }
  const ctype = req.headers.get("content-type") ?? "";
  if (!ctype.includes("multipart/form-data")) {
    return domainResponse({ kind: "MissingMultipart" });
  }
  const buffers = await loadCompareBuffers(await req.formData());
  if (!buffers.ok) return domainResponse(buffers.error);
  const jobId = newJobId();
  const key = jobIdToString(jobId);
  const r2a = `jobs/${key}/a`;
  const r2b = `jobs/${key}/b`;
  // Cola solo con IDs+punteros, nunca bytes (limite 128KB/mensaje).
  await env.VULTUS_BUCKET.put(r2a, buffers.value.aBuf);
  await env.VULTUS_BUCKET.put(r2b, buffers.value.bBuf);
  await env.VULTUS_QUEUE.send({ job_id: key, r2_keys: { image_a: r2a, image_b: r2b } });
  const ttl: TtlSecs = parseTtlSecsBranded(env.R2_TTL_SECONDS);
  const stub = env.VULTUS_PROGRESS.get(env.VULTUS_PROGRESS.idFromName(key));
  await stub.fetch(`https://do/init?job_id=${key}&ttl_secs=${ttlToNumber(ttl)}`);
  return json({ job_id: key, status: "queued" }, 202);
}

async function handleResult(id: JobId, env: Env): Promise<Response> {
  if (!env.VULTUS_BUCKET || !env.VULTUS_PROGRESS) {
    return reportAppError({
      kind: "MissingBindings",
      cause: "result requires VULTUS_BUCKET and VULTUS_PROGRESS",
    });
  }
  const key = jobIdToString(id);
  // Fuente de verdad: DO /status. 404 si nunca existio o purgo.
  // Si no esta done (queued/processing/expired/failed) => 409 para no esperar en vano.
  // Si done, R2 jobs/{id}/result.zip; 404 si falta el objeto.
  const stub = env.VULTUS_PROGRESS.get(env.VULTUS_PROGRESS.idFromName(key));
  const res = await stub.fetch("https://do/status");
  if (res.status === 404) {
    return domainResponse({ kind: "NotFound" });
  }
  const status = await readDoStatus(res);
  if (!status.ok) return reportAppError(status.error);
  if (status.value.status !== "done") {
    return domainResponse({ kind: "NotDone" });
  }
  const obj = await env.VULTUS_BUCKET.get(`jobs/${key}/result.zip`);
  if (obj && obj.body) {
    return new Response(obj.body, {
      headers: {
        ...CORS_HEADERS,
        "Content-Type": "application/zip",
        "Content-Disposition": `attachment; filename="result-${key}.zip"`,
      },
    });
  }
  return domainResponse({ kind: "NotFound" });
}

async function handleJob(id: JobId, env: Env): Promise<Response> {
  if (!env.VULTUS_PROGRESS) {
    return reportAppError({ kind: "MissingBindings", cause: "job requires VULTUS_PROGRESS" });
  }
  // El DO es fuente de verdad en edge. Lee estado real y 404 si nunca existió o ya purgó.
  const stub = env.VULTUS_PROGRESS.get(env.VULTUS_PROGRESS.idFromName(jobIdToString(id)));
  const res = await stub.fetch("https://do/status");
  if (res.status === 404) {
    return domainResponse({ kind: "NotFound" });
  }
  const status = await readDoStatus(res);
  if (!status.ok) return reportAppError(status.error);
  return json({ job_id: status.value.jobId, status: status.value.status });
}

async function handleProgress(id: JobId, req: Request, env: Env): Promise<Response> {
  if (!env.VULTUS_PROGRESS) {
    return reportAppError({ kind: "MissingBindings", cause: "progress requires VULTUS_PROGRESS" });
  }
  // Actualizacion de progreso desde el pull consumer Modal. Mismo seam HTTP,
  // sin seam nuevo: reenvia al DO /progress que valida 0..1 y stage.
  let payload: unknown;
  try {
    payload = await req.json();
  } catch {
    return domainResponse({ kind: "InvalidJson" });
  }
  const update = parseProgressUpdate(payload);
  if (!update.ok) return domainResponse(update.error);
  const stub = env.VULTUS_PROGRESS.get(env.VULTUS_PROGRESS.idFromName(jobIdToString(id)));
  return stub.fetch(
    new Request("https://do/progress", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: progressUpdateBody(update.value, payload),
    }),
  );
}

async function handleEvents(id: JobId, req: Request, env: Env): Promise<Response> {
  if (!env.VULTUS_PROGRESS) {
    return reportAppError({ kind: "MissingBindings", cause: "events requires VULTUS_PROGRESS" });
  }
  const stub = env.VULTUS_PROGRESS.get(env.VULTUS_PROGRESS.idFromName(jobIdToString(id)));
  return stub.fetch(req);
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    const { pathname } = url;

    // Preflight del navegador: 204 sin cuerpo, solo headers CORS.
    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: { ...CORS_HEADERS } });
    }

    if (pathname === "/health" && req.method === "GET") {
      const ttl: TtlSecs = parseTtlSecsBranded(env.R2_TTL_SECONDS);
      return Response.json(
        {
          status: "ok",
          gateway: "worker",
          queue: "ok",
          ttl_secs: ttlToNumber(ttl),
        },
        { headers: { ...CORS_HEADERS } },
      );
    }

    if (pathname === "/v1/compare" && req.method === "POST") {
      return handleCompare(req, env);
    }

    const resultMatch = pathname.match(/^\/v1\/jobs\/([^/]+)\/result$/);
    if (resultMatch && req.method === "GET") {
      const id = routeJobId(resultMatch[1]);
      if (!id.ok) return domainResponse(id.error);
      return handleResult(id.value, env);
    }

    const jobMatch = pathname.match(/^\/v1\/jobs\/([^/]+)$/);
    if (jobMatch && req.method === "GET") {
      const id = routeJobId(jobMatch[1]);
      if (!id.ok) return domainResponse(id.error);
      return handleJob(id.value, env);
    }

    const progressMatch = pathname.match(/^\/v1\/jobs\/([^/]+)\/progress$/);
    if (progressMatch && req.method === "POST") {
      const id = routeJobId(progressMatch[1]);
      if (!id.ok) return domainResponse(id.error);
      return handleProgress(id.value, req, env);
    }

    const evMatch = pathname.match(/^\/v1\/jobs\/([^/]+)\/events$/);
    if (evMatch) {
      const id = routeJobId(evMatch[1]);
      if (!id.ok) return domainResponse(id.error);
      return handleEvents(id.value, req, env);
    }

    return domainResponse({ kind: "NotFound" });
  },
};
