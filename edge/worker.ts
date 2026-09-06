/**
 * Gateway fino Cloudflare (prod vivo).
 * La API pesada vive en Rust Axum para paridad local; en prod el trafico
 * va edge -> Queues+R2 -> Modal workers -> R2 -> edge.
 * Pre-validacion espejo de `ImageBytes::parse` (size + magic) vía `contract.ts`.
 * En prod exige bindings reales (R2 + Queue + DO); sin fallbacks dummy.
 */
import {
  MAX_IMAGE_BYTES,
  hasSupportedMagic,
  isUuid,
  isValidProgress,
  isValidStage,
  parseTtlSecs,
} from "./contract";

interface Env {
  VULTUS_QUEUE: Queue;
  VULTUS_BUCKET: R2Bucket;
  VULTUS_PROGRESS: DurableObjectNamespace;
  QUEUE_DRIVER?: string;
  R2_TTL_SECONDS?: string;
}

export { ProgressDO } from "./progress-do";

// CORS permisivo como el `CorsLayer::permissive` de la API Rust local:
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

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    const { pathname } = url;

    // Preflight del navegador: 204 sin cuerpo, solo headers CORS.
    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: { ...CORS_HEADERS } });
    }

    if (pathname === "/health" && req.method === "GET") {
      return Response.json(
        {
          status: "ok",
          queue: "ok",
          ttl_secs: parseTtlSecs(env.R2_TTL_SECONDS),
        },
        { headers: { ...CORS_HEADERS } },
      );
    }

    if (pathname === "/v1/compare" && req.method === "POST") {
      if (!env.VULTUS_BUCKET || !env.VULTUS_QUEUE || !env.VULTUS_PROGRESS) {
        return json({ detail: "missing bindings" }, 500);
      }
      const ctype = req.headers.get("content-type") ?? "";
      if (!ctype.includes("multipart/form-data")) {
        return json({ detail: "missing multipart" }, 400);
      }
      const form = await req.formData();
      const a = form.get("image_a");
      const b = form.get("image_b");
      if (!(a instanceof File) || !(b instanceof File)) {
        return json({ detail: "missing image_a or image_b" }, 400);
      }
      if (a.size === 0 || b.size === 0 || a.size > MAX_IMAGE_BYTES || b.size > MAX_IMAGE_BYTES) {
        return json({ detail: "invalid image: size out of range" }, 400);
      }
      // Paridad con Rust `is_jpeg` / `is_png`: leer una vez y reusar para R2.
      const aBuf = await a.arrayBuffer();
      const bBuf = await b.arrayBuffer();
      if (!hasSupportedMagic(new Uint8Array(aBuf)) || !hasSupportedMagic(new Uint8Array(bBuf))) {
        return json({ detail: "invalid image: not jpeg nor png" }, 400);
      }
      const job_id = crypto.randomUUID();
      const r2a = `jobs/${job_id}/a`;
      const r2b = `jobs/${job_id}/b`;
      // Cola solo con IDs+punteros, nunca bytes (limite 128KB/mensaje).
      await env.VULTUS_BUCKET.put(r2a, aBuf);
      await env.VULTUS_BUCKET.put(r2b, bBuf);
      await env.VULTUS_QUEUE.send({ job_id, r2_keys: { image_a: r2a, image_b: r2b } });
      const ttlSecs = parseTtlSecs(env.R2_TTL_SECONDS);
      const doId = env.VULTUS_PROGRESS.idFromName(job_id);
      const stub = env.VULTUS_PROGRESS.get(doId);
      await stub.fetch(`https://do/init?job_id=${job_id}&ttl_secs=${ttlSecs}`);
      return json({ job_id, status: "queued" }, 202);
    }

    const resultMatch = pathname.match(/^\/v1\/jobs\/([^/]+)\/result$/);
    if (resultMatch && req.method === "GET") {
      const id = resultMatch[1];
      if (!isUuid(id)) {
        return json({ detail: "invalid job_id" }, 400);
      }
      if (!env.VULTUS_BUCKET || !env.VULTUS_PROGRESS) {
        return json({ detail: "missing bindings" }, 500);
      }
      // Fuente de verdad: DO /status. 404 si nunca existio o purgo.
      // Si no esta done (queued/processing/expired/failed) => 409 para no esperar en vano.
      // Si done, R2 jobs/{id}/result.zip; 404 si falta el objeto.
      const doId = env.VULTUS_PROGRESS.idFromName(id);
      const stub = env.VULTUS_PROGRESS.get(doId);
      const res = await stub.fetch(`https://do/status`);
      if (res.status === 404) {
        return json({ detail: "not found" }, 404);
      }
      const body = (await res.json()) as { job_id?: unknown; status?: unknown };
      if (typeof body.status !== "string") {
        return json({ detail: "not found" }, 404);
      }
      if (body.status !== "done") {
        return json({ detail: "not done" }, 409);
      }
      const obj = await env.VULTUS_BUCKET.get(`jobs/${id}/result.zip`);
      if (obj && obj.body) {
        return new Response(obj.body, {
          headers: {
            ...CORS_HEADERS,
            "Content-Type": "application/zip",
            "Content-Disposition": `attachment; filename="result-${id}.zip"`,
          },
        });
      }
      return json({ detail: "not found" }, 404);
    }

    const jobMatch = pathname.match(/^\/v1\/jobs\/([^/]+)$/);
    if (jobMatch && req.method === "GET") {
      const id = jobMatch[1];
      if (!isUuid(id)) {
        return json({ detail: "invalid job_id" }, 400);
      }
      if (!env.VULTUS_PROGRESS) {
        return json({ detail: "missing bindings" }, 500);
      }
      // El DO es fuente de verdad en edge. Lee estado real y 404 si nunca existió o ya purgó.
      const doId = env.VULTUS_PROGRESS.idFromName(id);
      const stub = env.VULTUS_PROGRESS.get(doId);
      const res = await stub.fetch(`https://do/status`);
      const body = (await res.json()) as { job_id?: unknown; status?: unknown; detail?: unknown };
      if (res.status === 404) {
        return json({ detail: "not found" }, 404);
      }
      if (typeof body.job_id === "string" && typeof body.status === "string") {
        return json({ job_id: body.job_id, status: body.status });
      }
      return json({ detail: "not found" }, 404);
    }

    const progressMatch = pathname.match(/^\/v1\/jobs\/([^/]+)\/progress$/);
    if (progressMatch && req.method === "POST") {
      const id = progressMatch[1];
      if (!isUuid(id)) {
        return json({ detail: "invalid job_id" }, 400);
      }
      if (!env.VULTUS_PROGRESS) {
        return json({ detail: "missing bindings" }, 500);
      }
      // Actualizacion de progreso desde el pull consumer Modal. Mismo seam HTTP,
      // sin seam nuevo: reenvia al DO /progress que valida 0..1 y stage.
      let payload: unknown;
      try {
        payload = await req.json();
      } catch {
        return json({ detail: "invalid json" }, 400);
      }
      const p = payload as { progress?: unknown; stage?: unknown; status?: unknown };
      if (p.progress !== undefined && !isValidProgress(p.progress)) {
        return json({ detail: "invalid progress" }, 400);
      }
      if (p.stage !== undefined && !isValidStage(p.stage)) {
        return json({ detail: "invalid stage" }, 400);
      }
      const doId = env.VULTUS_PROGRESS.idFromName(id);
      const stub = env.VULTUS_PROGRESS.get(doId);
      return stub.fetch(
        new Request(`https://do/progress`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
      );
    }

    const evMatch = pathname.match(/^\/v1\/jobs\/([^/]+)\/events$/);
    if (evMatch) {
      const id = evMatch[1];
      if (!isUuid(id)) {
        return json({ detail: "invalid job_id" }, 400);
      }
      if (!env.VULTUS_PROGRESS) {
        return json({ detail: "missing bindings" }, 500);
      }
      const doId = env.VULTUS_PROGRESS.idFromName(id);
      const stub = env.VULTUS_PROGRESS.get(doId);
      return stub.fetch(req);
    }

    return json({ detail: "not found" }, 404);
  },
};
