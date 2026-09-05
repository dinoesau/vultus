/**
 * Durable Object de progreso (Fase 0).
 * Espejo de `Store` en Rust: TTL logico parametrizado, ventana `Expired`
 * visible hasta 2x TTL y luego purga. Sin persistencia mas alla del TTL.
 */
import {
  RESULT_TTL_SECONDS,
  isValidProgress,
  isValidStage,
  parseTtlSecs,
} from "./contract";

export class ProgressDO {
  state: DurableObjectState;
  progress = 0;
  stage = "queued";
  status = "queued";
  job_id = "unknown";
  ttlSecs = RESULT_TTL_SECONDS;

  constructor(state: DurableObjectState) {
    this.state = state;
  }

  private async load(): Promise<void> {
    const stored = await this.state.storage.get<Record<string, unknown>>([
      "job_id",
      "progress",
      "stage",
      "status",
      "ttl_secs",
    ]);
    // `get` con array retorna Map en runtime Cloudflare.
    const get = (k: string): unknown =>
      stored instanceof Map ? stored.get(k) : (stored as Record<string, unknown>)?.[k];
    const jobId = get("job_id");
    const progress = get("progress");
    const stage = get("stage");
    const status = get("status");
    const ttl = get("ttl_secs");
    if (typeof jobId === "string") this.job_id = jobId;
    if (typeof progress === "number") this.progress = progress;
    if (typeof stage === "string") this.stage = stage;
    if (typeof status === "string") this.status = status;
    if (typeof ttl === "number") this.ttlSecs = ttl;
  }

  private async save(): Promise<void> {
    await this.state.storage.put({
      job_id: this.job_id,
      progress: this.progress,
      stage: this.stage,
      status: this.status,
      ttl_secs: this.ttlSecs,
    });
  }

  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/init") {
      this.job_id = url.searchParams.get("job_id") ?? "unknown";
      this.ttlSecs = parseTtlSecs(url.searchParams.get("ttl_secs"));
      this.progress = 0;
      this.stage = "queued";
      this.status = "queued";
      await this.save();
      // TTL logico: a los TTL marcamos expired; a los 2x TTL purgamos.
      await this.state.storage.setAlarm(Date.now() + this.ttlSecs * 1000);
      return Response.json({ ok: true, job_id: this.job_id, ttl_secs: this.ttlSecs });
    }
    if (url.pathname === "/progress" && req.method === "POST") {
      const body = (await req.json()) as { progress?: unknown; stage?: unknown };
      if (body.progress !== undefined && !isValidProgress(body.progress)) {
        return Response.json({ detail: "invalid progress" }, { status: 400 });
      }
      if (body.stage !== undefined && !isValidStage(body.stage)) {
        return Response.json({ detail: "invalid stage" }, { status: 400 });
      }
      await this.load();
      if (typeof body.progress === "number") this.progress = body.progress;
      if (typeof body.stage === "string") {
        this.stage = body.stage;
        this.status = body.stage === "done" ? "done" : "processing";
      }
      await this.save();
      return Response.json({ ok: true });
    }
    await this.load();
    if (url.pathname === "/status" && req.method === "GET") {
      if (this.job_id === "unknown") {
        return Response.json({ detail: "not found" }, { status: 404 });
      }
      return Response.json({ job_id: this.job_id, status: this.status });
    }
    if (this.job_id === "unknown") {
      return Response.json({ detail: "not found" }, { status: 404 });
    }
    const upgrade = req.headers.get("Upgrade");
    if (upgrade !== "websocket") {
      return new Response("expected websocket", { status: 400 });
    }
    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair) as [WebSocket, WebSocket];
    server.accept();
    server.send(
      JSON.stringify({
        job_id: this.job_id,
        progress: this.progress,
        stage: this.stage,
        status: this.status,
      }),
    );
    server.close(1000, "fase0 snapshot");
    return new Response(null, { status: 101, webSocket: client });
  }

  async alarm(): Promise<void> {
    await this.load();
    if (this.status !== "expired") {
      // Primera alarma (TTL): ventana visible como expired, como Rust `Expired`.
      this.status = "expired";
      await this.save();
      await this.state.storage.setAlarm(Date.now() + this.ttlSecs * 1000);
      return;
    }
    // Segunda alarma (2x TTL): purga total, como `Store::purge_expired`.
    await this.state.storage.deleteAll();
    this.job_id = "unknown";
    this.progress = 0;
    this.stage = "queued";
    this.status = "queued";
    this.ttlSecs = RESULT_TTL_SECONDS;
  }
}
