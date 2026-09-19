/**
 * Entrada dev del gateway unico (nunca prod).
 * Envuelve el handler de produccion (`worker.ts`) y agrega exactamente
 * dos rutas dev: lectura de blobs de entrada y escritura del bundle.
 * Las rutas dev solo responden con `ALLOW_DEV_ROUTES=1`; con vars de
 * produccion responden 404. El bundle de prod no referencia este modulo.
 */
import { jobIdToString, parseJobId } from "./contract";
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

type Slot = "a" | "b";
type SlotError = { readonly kind: "InvalidSlot" };
type ZipError = { readonly kind: "InvalidBundle" };

/** Parse-once: slot a/b como Result, sin boolean isSlot aguas abajo. */
function parseSlot(raw: unknown): import("./contract").Result<Slot, SlotError> {
  if (raw === "a" || raw === "b") return { ok: true, value: raw };
  return { ok: false, error: { kind: "InvalidSlot" } };
}

/** Parse-once: bundle zip (PK) como Result, sin boolean isZip aguas abajo. */
function parseZipBundle(raw: unknown): import("./contract").Result<Uint8Array, ZipError> {
  if (!(raw instanceof Uint8Array)) return { ok: false, error: { kind: "InvalidBundle" } };
  const b0 = raw[0];
  const b1 = raw[1];
  if (raw.length === 0 || raw.length > DEV_MAX_RESULT_BYTES || b0 !== 0x50 || b1 !== 0x4b) {
    return { ok: false, error: { kind: "InvalidBundle" } };
  }
  return { ok: true, value: raw };
}

export default {
  async fetch(req: Request, env: DevEnv): Promise<Response> {
    const url = new URL(req.url);
    const { pathname } = url;

    const blobMatch = pathname.match(/^\/dev\/blobs\/([^/]+)\/([^/]+)$/);
    if (blobMatch && req.method === "GET") {
      if (!devEnabled(env)) return json({ detail: "not found" }, 404);
      const jobRaw: unknown = blobMatch[1];
      const slotRaw: unknown = blobMatch[2];
      const parsed = parseJobId(jobRaw);
      if (!parsed.ok) return json({ detail: "invalid job_id" }, 400);
      const slot = parseSlot(slotRaw);
      if (!slot.ok) return json({ detail: "invalid slot" }, 400);
      if (!env.VULTUS_BUCKET) return json({ detail: "missing bindings" }, 500);
      // Llave via accesor del brand probado, sin `as`.
      const obj = await env.VULTUS_BUCKET.get(`jobs/${jobIdToString(parsed.value)}/${slot.value}`);
      if (!obj || !obj.body) return json({ detail: "not found" }, 404);
      return new Response(obj.body, {
        headers: { "Content-Type": "application/octet-stream" },
      });
    }

    const resultMatch = pathname.match(/^\/dev\/results\/([^/]+)$/);
    if (resultMatch && req.method === "PUT") {
      if (!devEnabled(env)) return json({ detail: "not found" }, 404);
      const jobRaw: unknown = resultMatch[1];
      const parsed = parseJobId(jobRaw);
      if (!parsed.ok) return json({ detail: "invalid job_id" }, 400);
      if (!env.VULTUS_BUCKET) return json({ detail: "missing bindings" }, 500);
      const bundle = parseZipBundle(new Uint8Array(await req.arrayBuffer()));
      if (!bundle.ok) return json({ detail: "invalid result bundle" }, 400);
      await env.VULTUS_BUCKET.put(`jobs/${jobIdToString(parsed.value)}/result.zip`, bundle.value);
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
