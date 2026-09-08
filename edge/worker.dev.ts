/**
 * Entrada dev del gateway unico (nunca prod).
 * Envuelve el handler de produccion (`worker.ts`) y agrega exactamente
 * dos rutas dev: lectura de blobs de entrada y escritura del bundle.
 * Las rutas dev solo responden con `ALLOW_DEV_ROUTES=1`; con vars de
 * produccion responden 404. El bundle de prod no referencia este modulo.
 */
import { parseJobId } from "./contract";
import prod from "./worker";

export { ProgressDO } from "./progress-do";

interface DevEnv {
  VULTUS_BUCKET?: R2Bucket;
  VULTUS_QUEUE?: Queue;
  VULTUS_PROGRESS?: DurableObjectNamespace;
  ALLOW_DEV_ROUTES?: string;
  RUNNER_WEBHOOK_URL?: string;
}

const DEV_MAX_RESULT_BYTES = 32 * 1024 * 1024;

function json(data: unknown, status = 200): Response {
  return Response.json(data, { status });
}

function devEnabled(env: DevEnv): boolean {
  return env.ALLOW_DEV_ROUTES === "1";
}

function isSlot(s: string): s is "a" | "b" {
  return s === "a" || s === "b";
}

function isZip(bytes: Uint8Array): boolean {
  return bytes.length >= 4 && bytes[0] === 0x50 && bytes[1] === 0x4b;
}

export default {
  async fetch(req: Request, env: DevEnv): Promise<Response> {
    const url = new URL(req.url);
    const { pathname } = url;

    const blobMatch = pathname.match(/^\/dev\/blobs\/([^/]+)\/([^/]+)$/);
    if (blobMatch && req.method === "GET") {
      if (!devEnabled(env)) return json({ detail: "not found" }, 404);
      const parsed = parseJobId(blobMatch[1]);
      if (!parsed.ok) return json({ detail: "invalid job_id" }, 400);
      if (!isSlot(blobMatch[2])) return json({ detail: "invalid slot" }, 400);
      if (!env.VULTUS_BUCKET) return json({ detail: "missing bindings" }, 500);
      const obj = await env.VULTUS_BUCKET.get(`jobs/${blobMatch[1]}/${blobMatch[2]}`);
      if (!obj || !obj.body) return json({ detail: "not found" }, 404);
      return new Response(obj.body, {
        headers: { "Content-Type": "application/octet-stream" },
      });
    }

    const resultMatch = pathname.match(/^\/dev\/results\/([^/]+)$/);
    if (resultMatch && req.method === "PUT") {
      if (!devEnabled(env)) return json({ detail: "not found" }, 404);
      const parsed = parseJobId(resultMatch[1]);
      if (!parsed.ok) return json({ detail: "invalid job_id" }, 400);
      if (!env.VULTUS_BUCKET) return json({ detail: "missing bindings" }, 500);
      const buf = new Uint8Array(await req.arrayBuffer());
      if (buf.length === 0 || buf.length > DEV_MAX_RESULT_BYTES || !isZip(buf)) {
        return json({ detail: "invalid result bundle" }, 400);
      }
      await env.VULTUS_BUCKET.put(`jobs/${resultMatch[1]}/result.zip`, buf);
      return json({ ok: true });
    }

    return (prod as { fetch(req: Request, env: unknown): Promise<Response> }).fetch(req, env);
  },

  // Consumer dev: reenvia cada mensaje al runner local via webhook.
  // Sin el runner (falta RUNNER_WEBHOOK_URL) falla ruidoso, nunca silencioso.
  async queue(batch: { messages: Array<{ body: unknown; ack(): void; retry(): void }> }, env: DevEnv): Promise<void> {
    const target = (env.RUNNER_WEBHOOK_URL ?? "").trim();
    if (!target) throw new Error("missing RUNNER_WEBHOOK_URL for dev queue consumer");
    for (const msg of batch.messages) {
      let res: Response;
      try {
        res = await fetch(target, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(msg.body),
        });
      } catch {
        msg.retry();
        continue;
      }
      if (res.ok) msg.ack();
      else msg.retry();
    }
  },
};
