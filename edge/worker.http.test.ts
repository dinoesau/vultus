/**
 * Seam 1 HTTP en el runtime worker: port 1:1 de `backend/tests/test_api.py`.
 * Goldens literales, sin mocks internos; websocket y backdoor incluidos.
 */
import { SELF } from "cloudflare:test";
import { describe, expect, it } from "vitest";
import devEntry from "./worker.dev";
import { STAGES } from "./contract";

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
    const body = (await res.json()) as { status: string; gateway: string; queue: string; ttl_secs: number };
    expect(body.status).toBe("ok");
    expect(body.gateway).toBe("worker");
    expect(body.queue).toBe("ok");
    expect(body.ttl_secs).toBe(60);
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
});
