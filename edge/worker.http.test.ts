/**
 * Seam 1 HTTP en el runtime worker: port 1:1 de `backend/tests/test_api.py`.
 * Goldens literales, sin mocks internos; websocket y backdoor incluidos.
 * TTL estricto: health valido sin ttl_error, invalido via resolveTtlSecs unit
 * mas handler fake-env (sin mutar env compartido del pool); DO init 400 sin
 * store/alarma; paridad "60"; compare init !ok 502 sin R2/Queue; storage
 * corrupto revalida a 60 con log throttled y alarma 60s/120s.
 */
import { SELF, env } from "cloudflare:test";
import { describe, expect, it, vi } from "vitest";
import devEntry from "./worker.dev";
import worker from "./worker";
import { ProgressDO } from "./progress-do";
import { STAGES, resolveTtlSecs, ttlToNumber } from "./contract";

function png(): Uint8Array {
  return new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, ...new Array(56).fill(0)]);
}

function compareForm(): FormData {
  const fd = new FormData();
  fd.append("image_a", new File([png()], "a.png", { type: "image/png" }));
  fd.append("image_b", new File([png()], "b.png", { type: "image/png" }));
  return fd;
}

async function newJobId(): Promise<string> {
  const res = await SELF.fetch("https://gateway/v1/compare", { method: "POST", body: compareForm() });
  expect(res.status).toBe(202);
  const body = (await res.json()) as { job_id: string; status: string };
  expect(body.status).toBe("queued");
  return body.job_id;
}

type TestEnv = {
  VULTUS_BUCKET: R2Bucket;
  VULTUS_QUEUE: Queue;
  VULTUS_PROGRESS: DurableObjectNamespace;
};

function testEnv(): TestEnv {
  return env as unknown as TestEnv;
}

describe("gateway http en worker runtime", () => {
  it("post compare 202 queued luego get status", async () => {
    const job_id = await newJobId();
    const status = await SELF.fetch(`https://gateway/v1/jobs/${job_id}`);
    expect(status.status).toBe(200);
    const body = (await status.json()) as { job_id: string; status: string };
    expect(body.job_id).toBe(job_id);
    expect(["queued", "processing"]).toContain(body.status);
  });

  it("400 en imagen faltante o invalida y uuid roto", async () => {
    const onlyA = new FormData();
    onlyA.append("image_a", new File([png()], "a.png", { type: "image/png" }));
    const missing = await SELF.fetch("https://gateway/v1/compare", { method: "POST", body: onlyA });
    expect(missing.status).toBe(400);

    const bad = new FormData();
    bad.append("image_a", new File([new Uint8Array([1, 2, 3])], "a.bin", { type: "application/octet-stream" }));
    bad.append("image_b", new File([new Uint8Array([4, 5, 6])], "b.bin", { type: "application/octet-stream" }));
    const invalid = await SELF.fetch("https://gateway/v1/compare", { method: "POST", body: bad });
    expect(invalid.status).toBe(400);

    expect((await SELF.fetch("https://gateway/v1/jobs/not-a-uuid")).status).toBe(400);
  });

  it("404 en desconocido y 409 en result pre-done", async () => {
    const unknown = "11111111-1111-4111-8111-111111111111";
    expect((await SELF.fetch(`https://gateway/v1/jobs/${unknown}`)).status).toBe(404);
    expect((await SELF.fetch(`https://gateway/v1/jobs/${unknown}/result`)).status).toBe(404);
    const job_id = await newJobId();
    expect((await SELF.fetch(`https://gateway/v1/jobs/${job_id}/result`)).status).toBe(409);
  });

  it("health reporta gateway worker y ttl", async () => {
    const res = await SELF.fetch("https://gateway/health");
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      status: string;
      gateway: string;
      queue: string;
      ttl_secs: number;
      ttl_error?: string;
    };
    expect(body.status).toBe("ok");
    expect(body.gateway).toBe("worker");
    expect(body.queue).toBe("ok");
    expect(body.ttl_secs).toBe(60);
    expect("ttl_error" in body).toBe(false);
  });

  it("ws emite snapshot queued y handshake falla en desconocido", async () => {
    const job_id = await newJobId();
    const resp = await SELF.fetch(`https://gateway/v1/jobs/${job_id}/events`, {
      headers: { Upgrade: "websocket" },
    });
    const ws = (resp as unknown as { webSocket?: WebSocket }).webSocket;
    expect(ws).toBeDefined();
    if (!ws) return;
    ws.accept();
    const first = await new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("ws timeout")), 3000);
      ws.addEventListener("message", (ev) => {
        clearTimeout(timer);
        resolve(String((ev as MessageEvent).data));
      });
    });
    const event = JSON.parse(first) as { job_id: string; status: string; stage: string };
    expect(event.job_id).toBe(job_id);
    expect(["queued", "processing"]).toContain(event.status);
    expect([...STAGES]).toContain(event.stage);
    ws.close();
    await new Promise((r) => setTimeout(r, 800));

    const unknown = await SELF.fetch(
      "https://gateway/v1/jobs/11111111-1111-4111-8111-111111111111/events",
      { headers: { Upgrade: "websocket" } },
    );
    expect(unknown.status).toBe(404);
  });

  it("rutas dev responden 404 con vars de produccion", async () => {
    const prodLikeEnv = {};
    const blob = await devEntry.fetch(
      new Request("https://gateway/dev/blobs/11111111-1111-4111-8111-111111111111/a"),
      prodLikeEnv as never,
    );
    expect(blob.status).toBe(404);
    const result = await devEntry.fetch(
      new Request("https://gateway/dev/results/11111111-1111-4111-8111-111111111111", {
        method: "PUT",
        body: new Uint8Array([0x50, 0x4b, 0x03, 0x04]),
      }),
      prodLikeEnv as never,
    );
    expect(result.status).toBe(404);
  });

  it("resolveTtlSecs invalido usa default 60", () => {
    for (const raw of ["nope", "0", 0, 9999, "060", 60.9, undefined, null, "", "   "]) {
      const r = resolveTtlSecs(raw);
      expect(r.invalid).toBe(true);
      expect(ttlToNumber(r.ttl)).toBe(60);
    }
    const ok = resolveTtlSecs("60");
    expect(ok.invalid).toBe(false);
    expect(ttlToNumber(ok.ttl)).toBe(60);
  });

  it("salud invalida via handler fake-env reporta ttl_error sin log", async () => {
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const res = await worker.fetch(new Request("https://gateway/health"), {
        R2_TTL_SECONDS: "nope",
      } as never);
      expect(res.status).toBe(200);
      const body = (await res.json()) as { ttl_secs: number; ttl_error?: string };
      expect(body.ttl_secs).toBe(60);
      expect(body.ttl_error).toBe("InvalidTtlSecs"); // ttl_error presente en salud invalida
      expect(errSpy).toHaveBeenCalledTimes(0);
    } finally {
      errSpy.mockRestore();
    }
  });

  it("DO init ttl invalido 400 sin store ni alarma", async () => {
    // No-alarm por codigo: progress-do.ts /init retorna 400 antes de
    // save()/setAlarm (lineas 96-114); aqui se prueba via comportamiento
    // observable (GET /status -> 404 + R2.get == null). La prueba unitaria
    // fake-storage de abajo clava setAlarm times 0 sin tocar el env del pool.
    const te = testEnv();
    const cases: Array<{ ttlParam: string | null; label: string }> = [
      { ttlParam: "0", label: "cero" },
      { ttlParam: "nope", label: "basura" },
      { ttlParam: null, label: "ausente" },
    ];
    for (const c of cases) {
      const fresh = crypto.randomUUID();
      const stub = te.VULTUS_PROGRESS.get(te.VULTUS_PROGRESS.idFromName(fresh));
      const query =
        c.ttlParam === null
          ? `job_id=${fresh}`
          : `job_id=${fresh}&ttl_secs=${encodeURIComponent(c.ttlParam)}`;
      const initRes = await stub.fetch(`https://do/init?${query}`);
      expect(initRes.status).toBe(400);
      const statusRes = await stub.fetch("https://do/status");
      expect(statusRes.status).toBe(404);
      const r2obj = await te.VULTUS_BUCKET.get(`jobs/${fresh}/a`);
      expect(r2obj == null).toBe(true); // R2.get == null prueba no-store/no-alarm ttl invalido
    }
  });

  it("DO init ttl invalido 400 unit fake-storage sin setAlarm", async () => {
    const map = new Map<string, unknown>();
    const setAlarm = vi.fn(async (_t: number | Date) => {});
    const storage = {
      get: async (keys: string[]) => {
        const m = new Map<string, unknown>();
        for (const k of keys) if (map.has(k)) m.set(k, map.get(k));
        return m;
      },
      put: async (entries: Record<string, unknown>) => {
        for (const [k, v] of Object.entries(entries)) map.set(k, v);
      },
      setAlarm,
      deleteAll: async () => {
        map.clear();
      },
    } as unknown as DurableObjectStorage;
    const state = { storage } as unknown as DurableObjectState;
    const fake = new ProgressDO(state);
    const fresh = crypto.randomUUID();
    const initRes = await fake.fetch(new Request(`https://do/init?job_id=${fresh}&ttl_secs=nope`));
    expect(initRes.status).toBe(400);
    expect(setAlarm).toHaveBeenCalledTimes(0);
    expect(map.size).toBe(0);
    const statusRes = await fake.fetch(new Request("https://do/status"));
    expect(statusRes.status).toBe(404);
  });

  it("paridad handleCompare -> /init acepta '60' en ambos lados", async () => {
    const r = resolveTtlSecs("60");
    expect(r.invalid).toBe(false);
    expect(ttlToNumber(r.ttl)).toBe(60);
    const te = testEnv();
    const fresh = crypto.randomUUID();
    const stub = te.VULTUS_PROGRESS.get(te.VULTUS_PROGRESS.idFromName(fresh));
    const initRes = await stub.fetch(`https://do/init?job_id=${fresh}&ttl_secs=60`);
    expect(initRes.status).toBe(200);
    const body = (await initRes.json()) as { ok: boolean; ttl_secs: number };
    expect(body.ttl_secs).toBe(60);
    const statusRes = await stub.fetch("https://do/status");
    expect(statusRes.status).toBe(200);
    // Seam publico: POST /v1/compare con env default (60) -> 202 y el DO
    // queda iniciado (prueba que handleCompare cablea resolve -> /init).
    const cmp = await SELF.fetch("https://gateway/v1/compare", { method: "POST", body: compareForm() });
    expect(cmp.status).toBe(202);
    const cmpBody = (await cmp.json()) as { job_id: string };
    const cmpStub = te.VULTUS_PROGRESS.get(te.VULTUS_PROGRESS.idFromName(cmpBody.job_id));
    expect((await cmpStub.fetch("https://do/status")).status).toBe(200);
  });

  it("compare con init !ok va a 502 sin R2 ni Queue", async () => {
    const store = new Map<string, unknown>();
    const sendSpy = vi.fn(async (_msg: unknown) => {});
    const seen: string[] = [];
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const fakeBucket = {
        put: async (key: string, value: unknown) => {
          store.set(key, value);
        },
        get: async (key: string) => {
          const v = store.get(key);
          if (v === undefined) return null;
          return { body: v } as never;
        },
        delete: async (key: string) => {
          store.delete(key);
        },
      } as unknown as R2Bucket;
      const failingStub = {
        fetch: async (input: RequestInfo | URL) => {
          const raw = typeof input === "string" ? input : (input as Request).url;
          const url = new URL(raw);
          if (url.pathname === "/init") {
            return new Response(JSON.stringify({ detail: "upstream" }), { status: 500 });
          }
          return Response.json({ detail: "not found" }, { status: 404 });
        },
      } as unknown as DurableObjectStub;
      const fakeNs = {
        idFromName: (name: string) => {
          seen.push(name);
          return { toString: () => name } as unknown as DurableObjectId;
        },
        get: () => failingStub,
      } as unknown as DurableObjectNamespace;
      const fakeEnv = {
        VULTUS_BUCKET: fakeBucket,
        VULTUS_QUEUE: { send: sendSpy } as unknown as Queue,
        VULTUS_PROGRESS: fakeNs,
        R2_TTL_SECONDS: "60",
      };
      const res = await worker.fetch(
        new Request("https://gateway/v1/compare", { method: "POST", body: compareForm() }),
        fakeEnv as never,
      );
      expect(res.status).toBe(502);
      expect(seen.length).toBe(1);
      const gen = seen[0] as string;
      const status = await worker.fetch(
        new Request(`https://gateway/v1/jobs/${gen}`),
        fakeEnv as never,
      );
      expect(status.status).toBe(404);
      const ra = await fakeBucket.get(`jobs/${gen}/a`);
      expect(ra == null).toBe(true); // R2.get == null tras init !ok con cleanup
      const rb = await fakeBucket.get(`jobs/${gen}/b`);
      expect(rb == null).toBe(true); // R2.get == null tras init !ok con cleanup
      expect(sendSpy).toHaveBeenCalledTimes(0); // Queue.send == 0 on init !ok orphan-Queue documentado
      const initFailed = errSpy.mock.calls.filter(
        ([msg]) => typeof msg === "string" && (msg as string).includes("init failed"),
      );
      expect(initFailed.length).toBe(1);
      expect(errSpy).toHaveBeenCalledTimes(1);
    } finally {
      errSpy.mockRestore();
    }
  });

  it("compare Queue.send throw compensa R2 y va a 502", async () => {
    // DAG v9: handleCompare init ok + Queue.send throw compensa R2 (delete a|b)
    // y responde 502; el DO queued queda acotado por TTL 60s/120s (sin cancel).
    // Unit fake-env sin mutar env compartido del pool; job_id fresco via seen[0].
    const store = new Map<string, unknown>();
    const sendThrow = vi.fn(async (_msg: unknown) => {
      throw new Error("queue down");
    });
    const seen: string[] = [];
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const okStub = {
        fetch: async (input: RequestInfo | URL) => {
          const raw = typeof input === "string" ? input : (input as Request).url;
          const url = new URL(raw);
          if (url.pathname === "/init") {
            return Response.json({ ok: true, ttl_secs: 60 }, { status: 200 });
          }
          return Response.json({ detail: "not found" }, { status: 404 });
        },
      } as unknown as DurableObjectStub;
      const fakeNs = {
        idFromName: (name: string) => {
          seen.push(name);
          return { toString: () => name } as unknown as DurableObjectId;
        },
        get: () => okStub,
      } as unknown as DurableObjectNamespace;
      const fakeBucket = {
        put: async (key: string, value: unknown) => {
          store.set(key, value);
        },
        get: async (key: string) => {
          const v = store.get(key);
          if (v === undefined) return null;
          return { body: v } as never;
        },
        delete: async (key: string) => {
          store.delete(key);
        },
      } as unknown as R2Bucket;
      const fakeEnv = {
        VULTUS_BUCKET: fakeBucket,
        VULTUS_QUEUE: { send: sendThrow } as unknown as Queue,
        VULTUS_PROGRESS: fakeNs,
        R2_TTL_SECONDS: "60",
      };
      const res = await worker.fetch(
        new Request("https://gateway/v1/compare", { method: "POST", body: compareForm() }),
        fakeEnv as never,
      );
      expect(res.status).toBe(502);
      expect(await res.json()).toEqual({ detail: "upstream error" });
      expect(seen.length).toBe(1);
      expect(sendThrow).toHaveBeenCalledTimes(1);
      const gen = seen[0] as string;
      // Cleanup tras send throw: R2 ya no conserva a|b.
      const ra = await fakeBucket.get(`jobs/${gen}/a`);
      expect(ra == null).toBe(true);
      const rb = await fakeBucket.get(`jobs/${gen}/b`);
      expect(rb == null).toBe(true);
      const queueFailed = errSpy.mock.calls.filter(
        ([msg]) => typeof msg === "string" && (msg as string).includes("queue failed"),
      );
      expect(queueFailed.length).toBe(1);
    } finally {
      errSpy.mockRestore();
    }
  });

  it("storage corrupto revalida a 60 con log throttled y alarma 60s/120s", async () => {
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const nowSpy = vi.spyOn(Date, "now").mockReturnValue(1_000_000);
    const makeFake = () => {
      const map = new Map<string, unknown>();
      const setAlarm = vi.fn(async (_t: number | Date) => {});
      const storage = {
        get: async (keys: string[]) => {
          const m = new Map<string, unknown>();
          for (const k of keys) if (map.has(k)) m.set(k, map.get(k));
          return m;
        },
        put: async (entries: Record<string, unknown>) => {
          for (const [k, v] of Object.entries(entries)) map.set(k, v);
        },
        setAlarm,
        deleteAll: async () => {
          map.clear();
        },
      } as unknown as DurableObjectStorage;
      return { state: { storage } as unknown as DurableObjectState, map, setAlarm, storage };
    };
    type Loadable = { load(): Promise<void> };
    try {
      // white-box: inyeccion directa a storage requerida por Step 5; las
      // lecturas publicas /status de abajo reconcilian con el seam publico.
      // Mismo job mismo valor corrupto 0: 2x load deja 60 y un solo log.
      const f1 = makeFake();
      const do1 = new ProgressDO(f1.state);
      await f1.storage.put({ job_id: crypto.randomUUID(), status: "queued", ttl_secs: 0 });
      await (do1 as unknown as Loadable).load();
      expect(ttlToNumber(do1.ttlSecs)).toBe(60);
      await (do1 as unknown as Loadable).load();
      expect(ttlToNumber(do1.ttlSecs)).toBe(60);
      expect(errSpy).toHaveBeenCalledTimes(1);
      // Readback por seam publico: tras load corrupto el job sigue observable con ttl 60.
      expect((await do1.fetch(new Request("https://do/status"))).status).toBe(200);
      expect(ttlToNumber(do1.ttlSecs)).toBe(60);
      // Dos jobs mismo valor: dos instancias, dos logs.
      const f2 = makeFake();
      const do2 = new ProgressDO(f2.state);
      await f2.storage.put({ job_id: crypto.randomUUID(), status: "queued", ttl_secs: 0 });
      await (do2 as unknown as Loadable).load();
      expect(ttlToNumber(do2.ttlSecs)).toBe(60);
      expect(errSpy).toHaveBeenCalledTimes(2);
      // Clave ausente: default silencioso sin log extra.
      const f3 = makeFake();
      const do3 = new ProgressDO(f3.state);
      await f3.storage.put({ job_id: crypto.randomUUID(), status: "queued" });
      await (do3 as unknown as Loadable).load();
      expect(ttlToNumber(do3.ttlSecs)).toBe(60);
      expect(errSpy).toHaveBeenCalledTimes(2);
      // Cadena corrupta "nope" tambien revalida a 60 con un log nuevo.
      const f4 = makeFake();
      const do4 = new ProgressDO(f4.state);
      await f4.storage.put({ job_id: crypto.randomUUID(), status: "queued", ttl_secs: "nope" });
      await (do4 as unknown as Loadable).load();
      expect(ttlToNumber(do4.ttlSecs)).toBe(60);
      expect(errSpy).toHaveBeenCalledTimes(3);
      errSpy.mockClear();
      // Alarma con TTL valido 60: init arma a 60s, primera alarma re-arma a 120s.
      const fv = makeFake();
      const dov = new ProgressDO(fv.state);
      const vId = crypto.randomUUID();
      const initRes = await dov.fetch(new Request(`https://do/init?job_id=${vId}&ttl_secs=60`));
      expect(initRes.status).toBe(200);
      expect(fv.setAlarm).toHaveBeenCalledTimes(1);
      expect(fv.setAlarm).toHaveBeenCalledWith(1_000_000 + 60 * 1000);
      nowSpy.mockReturnValue(1_000_000 + 60 * 1000);
      await dov.alarm();
      expect(fv.setAlarm).toHaveBeenCalledTimes(2);
      expect(fv.setAlarm).toHaveBeenLastCalledWith(1_000_000 + 120 * 1000);
      // Alarma con corrupto usa default 60, sin reloj real.
      const fc = makeFake();
      const doc = new ProgressDO(fc.state);
      await fc.storage.put({ job_id: crypto.randomUUID(), status: "queued", ttl_secs: 0 });
      nowSpy.mockReturnValue(2_000_000);
      fc.setAlarm.mockClear();
      errSpy.mockClear();
      await doc.alarm();
      expect(ttlToNumber(doc.ttlSecs)).toBe(60);
      expect(fc.setAlarm).toHaveBeenCalledWith(2_000_000 + 60 * 1000);
      expect(errSpy).toHaveBeenCalledTimes(1);
    } finally {
      errSpy.mockRestore();
      nowSpy.mockRestore();
    }
  });
});
