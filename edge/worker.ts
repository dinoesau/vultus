/**
 * Gateway fino Cloudflare (Fase 0).
 * La API pesada vive en Rust Axum; aqui solo: health, compare, status, WS via DO.
 * Pre-validacion espejo de `ImageBytes::parse` (size + magic) vía `contract.ts`.
 * En prod hace R2 PutObject + Queues enqueue {job_id, r2_keys}; en `wrangler dev`
 * corre sin R2 real (fallback en memoria) para paridad local.
 */
import {
  MAX_IMAGE_BYTES,
  hasSupportedMagic,
  isUuid,
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

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    const { pathname } = url;

    if (pathname === "/health" && req.method === "GET") {
      return Response.json({
        status: "ok",
        queue: "ok",
        ttl_secs: parseTtlSecs(env.R2_TTL_SECONDS),
      });
    }

    if (pathname === "/v1/compare" && req.method === "POST") {
      const ctype = req.headers.get("content-type") ?? "";
      if (!ctype.includes("multipart/form-data")) {
        return Response.json({ detail: "missing multipart" }, { status: 400 });
      }
      const form = await req.formData();
      const a = form.get("image_a");
      const b = form.get("image_b");
      if (!(a instanceof File) || !(b instanceof File)) {
        return Response.json({ detail: "missing image_a or image_b" }, { status: 400 });
      }
      if (a.size === 0 || b.size === 0 || a.size > MAX_IMAGE_BYTES || b.size > MAX_IMAGE_BYTES) {
        return Response.json({ detail: "invalid image: size out of range" }, { status: 400 });
      }
      // Paridad con Rust `is_jpeg` / `is_png`: leer una vez y reusar para R2.
      const aBuf = await a.arrayBuffer();
      const bBuf = await b.arrayBuffer();
      if (!hasSupportedMagic(new Uint8Array(aBuf)) || !hasSupportedMagic(new Uint8Array(bBuf))) {
        return Response.json({ detail: "invalid image: not jpeg nor png" }, { status: 400 });
      }
      const job_id = crypto.randomUUID();
      const r2a = `jobs/${job_id}/a`;
      const r2b = `jobs/${job_id}/b`;
      try {
        await env.VULTUS_BUCKET.put(r2a, aBuf);
        await env.VULTUS_BUCKET.put(r2b, bBuf);
        await env.VULTUS_QUEUE.send({ job_id, r2_keys: { image_a: r2a, image_b: r2b } });
      } catch {
        // `wrangler dev` sin bindings reales: seguimos como dummy trazable.
      }
      const ttlSecs = parseTtlSecs(env.R2_TTL_SECONDS);
      const doId = env.VULTUS_PROGRESS.idFromName(job_id);
      const stub = env.VULTUS_PROGRESS.get(doId);
      await stub.fetch(`https://do/init?job_id=${job_id}&ttl_secs=${ttlSecs}`).catch(() => undefined);
      return Response.json({ job_id, status: "queued" }, { status: 202 });
    }

    const resultMatch = pathname.match(/^\/v1\/jobs\/([^/]+)\/result$/);
    if (resultMatch && req.method === "GET") {
      const id = resultMatch[1];
      if (!isUuid(id)) {
        return Response.json({ detail: "invalid job_id" }, { status: 400 });
      }
      // Fuente de verdad: DO /status como en jobStatus. 404 si nunca existio o purgo.
      // Si no esta done (queued/processing/expired) => 409 para no esperar en vano.
      // Si done, intenta R2 jobs/{id}/result.zip; si no hay binding (wrangler dev) => 404 trazable.
      try {
        const doId = env.VULTUS_PROGRESS.idFromName(id);
        const stub = env.VULTUS_PROGRESS.get(doId);
        const res = await stub.fetch(`https://do/status`);
        if (res.status === 404) {
          return Response.json({ detail: "not found" }, { status: 404 });
        }
        const body = (await res.json()) as { job_id?: unknown; status?: unknown };
        if (typeof body.status !== "string") {
          return Response.json({ detail: "not found" }, { status: 404 });
        }
        if (body.status !== "done") {
          return Response.json({ detail: "not done" }, { status: 409 });
        }
        try {
          // R2 directo; si no hay binding (wrangler dev) cae al catch => 404 trazable.
          const obj = await env.VULTUS_BUCKET.get(`jobs/${id}/result.zip`);
          if (obj && obj.body) {
            return new Response(obj.body, {
              headers: {
                "Content-Type": "application/zip",
                "Content-Disposition": `attachment; filename="result-${id}.zip"`,
              },
            });
          }
          return Response.json({ detail: "not found" }, { status: 404 });
        } catch {
          return Response.json({ detail: "not found" }, { status: 404 });
        }
      } catch {
        // `wrangler dev` sin binding DO: fallback trazable para no romper smoke local.
        return Response.json({ detail: "not found" }, { status: 404 });
      }
    }

    const jobMatch = pathname.match(/^\/v1\/jobs\/([^/]+)$/);
    if (jobMatch && req.method === "GET") {
      const id = jobMatch[1];
      if (!isUuid(id)) {
        return Response.json({ detail: "invalid job_id" }, { status: 400 });
      }
      // Decisión Fase 0 (ADR-007): el DO es fuente de verdad en edge, no dummy.
      // Lee estado real (queued/processing/expired) y 404 si nunca existió o ya purgó.
      try {
        const doId = env.VULTUS_PROGRESS.idFromName(id);
        const stub = env.VULTUS_PROGRESS.get(doId);
        const res = await stub.fetch(`https://do/status`);
        const body = (await res.json()) as { job_id?: unknown; status?: unknown; detail?: unknown };
        if (res.status === 404) {
          return Response.json({ detail: "not found" }, { status: 404 });
        }
        if (typeof body.job_id === "string" && typeof body.status === "string") {
          return Response.json({ job_id: body.job_id, status: body.status });
        }
        return Response.json({ detail: "not found" }, { status: 404 });
      } catch {
        // `wrangler dev` sin binding DO: fallback trazable para no romper smoke local.
        return Response.json({ job_id: id, status: "queued" });
      }
    }

    const evMatch = pathname.match(/^\/v1\/jobs\/([^/]+)\/events$/);
    if (evMatch) {
      const id = evMatch[1];
      if (!isUuid(id)) {
        return Response.json({ detail: "invalid job_id" }, { status: 400 });
      }
      const doId = env.VULTUS_PROGRESS.idFromName(id);
      const stub = env.VULTUS_PROGRESS.get(doId);
      return stub.fetch(req);
    }

    return Response.json({ detail: "not found" }, { status: 404 });
  },
};
