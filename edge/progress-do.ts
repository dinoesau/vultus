/**
 * Durable Object de progreso (prod vivo).
 * Espejo de `Store` en Python: TTL logico parametrizado, ventana `Expired`
 * visible hasta 2x TTL y luego purga. Sin persistencia mas alla del TTL.
 * Progreso vivo: WS emite snapshot inicial y luego ticks cada 500ms hasta
 * terminal (done/failed/expired) o 60s, en vez de snapshot+close.
 */
import {
  RESULT_TTL_SECONDS,
  isTerminalStatus,
  parseProgress,
  parseStage,
  parseTtlSecs,
  progressToNumber,
} from "./contract";

function isRecord(raw: unknown): raw is Record<string, unknown> {
  return typeof raw === "object" && raw !== null && !Array.isArray(raw);
}

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
    // `get` con array retorna Map en runtime Cloudflare: estrechar sin `as`.
    const get = (k: string): unknown => {
      if (stored instanceof Map) return stored.get(k);
      if (isRecord(stored)) return stored[k];
      return undefined;
    };
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
      let rawBody: unknown;
      try {
        rawBody = await req.json();
      } catch {
        return Response.json({ detail: "invalid json" }, { status: 400 });
      }
      const body: Record<string, unknown> = isRecord(rawBody) ? rawBody : {};
      // Parse-once via smart constructors; el DO guarda numeros/strings probados.
      if (body["progress"] !== undefined && !parseProgress(body["progress"]).ok) {
        return Response.json({ detail: "invalid progress" }, { status: 400 });
      }
      if (body["stage"] !== undefined && !parseStage(body["stage"]).ok) {
        return Response.json({ detail: "invalid stage" }, { status: 400 });
      }
      if (
        body["status"] !== undefined &&
        body["status"] !== "processing" &&
        body["status"] !== "done" &&
        body["status"] !== "failed"
      ) {
        return Response.json({ detail: "invalid status" }, { status: 400 });
      }
      await this.load();
      // Espejo de Python `Store`: expirado no acepta mas escrituras (404).
      // Sin esto un job lento resucitaria expired->done pasada la ventana.
      if (this.status === "expired") {
        return Response.json({ detail: "expired" }, { status: 404 });
      }
      const progressParsed = body["progress"] === undefined ? null : parseProgress(body["progress"]);
      const stageParsed = body["stage"] === undefined ? null : parseStage(body["stage"]);
      // Brands leidos via accesores, sin `as`: el numero viaja probado al storage.
      if (progressParsed !== null && progressParsed.ok) this.progress = progressToNumber(progressParsed.value);
      if (stageParsed !== null && stageParsed.ok) {
        this.stage = stageParsed.value;
      }
      if (body["status"] === "failed") {
        this.status = "failed";
      } else if (body["status"] === "done" || this.stage === "done") {
        this.status = "done";
      } else if (stageParsed !== null || progressParsed !== null) {
        if (this.status === "queued") this.status = "processing";
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
    // Estrechamiento explicito del par (runtime Cloudflare): sin `as`.
    // Un par malformado es bug de plataforma, no input de usuario.
    const parts: unknown[] = Object.values(pair);
    const client: unknown = parts[0];
    const server: unknown = parts[1];
    if (!(client instanceof WebSocket) || !(server instanceof WebSocket)) {
      return Response.json({ detail: "upstream error" }, { status: 502 });
    }
    server.accept();
    const snapshot = () =>
      JSON.stringify({
        job_id: this.job_id,
        progress: this.progress,
        stage: this.stage,
        status: this.status,
      });
    server.send(snapshot());
    // Progreso vivo: ticks 500ms hasta terminal o 60s, sin snapshot+close.
    // Cada tick recarga storage para ver updates del pull consumer Modal.
    const tickMs = 500;
    const maxTicks = 120;
    let ticks = 0;
    const timer = setInterval(async () => {
      ticks += 1;
      try {
        await this.load();
        try {
          server.send(snapshot());
        } catch {
          clearInterval(timer);
          return;
        }
        if (isTerminalStatus(this.status) || ticks >= maxTicks) {
          clearInterval(timer);
          try {
            server.close(1000, this.status);
          } catch {
            // Cierre best-effort: el cliente ya puede haberse ido.
          }
        }
      } catch {
        clearInterval(timer);
        try {
          server.close(1011, "load failed");
        } catch {
          // Cierre best-effort.
        }
      }
    }, tickMs);
    // Si el cliente se va, el runtime limpia el timer con el DO; no hay leaks mas alla del TTL.
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
    // Segunda alarma (2x TTL): purga total, como `Store::purge_expired` en Python.
    await this.state.storage.deleteAll();
    this.job_id = "unknown";
    this.progress = 0;
    this.stage = "queued";
    this.status = "queued";
    this.ttlSecs = RESULT_TTL_SECONDS;
  }
}
