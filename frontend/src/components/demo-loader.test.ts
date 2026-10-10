import { describe, expect, it } from "vitest";
import {
  DEMO_FILE_NAMES,
  DEMO_PATHS,
  blobsToFiles,
  demoImageToBlob,
  loadDemoBlobs,
  parseDemoBlob,
  resolveDemoPaths,
} from "./demo-loader.ts";

const JPEG_BYTES = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
const PNG_BYTES = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function jpegBlob(): Blob {
  return new Blob([JPEG_BYTES], { type: "image/jpeg" });
}

function demoResponse(blob: Blob, status = 200): Response {
  return new Response(blob, {
    status,
    headers: { "content-type": blob.type || "application/octet-stream" },
  });
}

describe("demo-loader: rutas demo (seam config)", () => {
  it("defaults apuntan a /demo/obama-*.jpg", () => {
    expect(DEMO_PATHS).toEqual({ a: "/demo/obama-a.jpg", b: "/demo/obama-b.jpg" });
    expect(resolveDemoPaths(undefined)).toEqual(DEMO_PATHS);
  });

  it("env VITE_DEMO_A/B_overridea, vacio se ignora", () => {
    expect(resolveDemoPaths({ VITE_DEMO_A: "/x/a.jpg" })).toEqual({
      a: "/x/a.jpg",
      b: "/demo/obama-b.jpg",
    });
    expect(resolveDemoPaths({ VITE_DEMO_A: "", VITE_DEMO_B: 42 })).toEqual(DEMO_PATHS);
    expect(() => resolveDemoPaths(null)).not.toThrow();
  });
});

describe("demo-loader: parseDemoBlob total con Result, nunca lanza", () => {
  it("JPEG y PNG validos dan Ok con bytes intactos", async () => {
    const parsed = await parseDemoBlob(jpegBlob());
    expect(parsed.ok).toBe(true);
    if (parsed.ok) {
      expect(await demoImageToBlob(parsed.value).arrayBuffer()).toEqual(
        await jpegBlob().arrayBuffer(),
      );
    }
    const png = await parseDemoBlob(new Blob([PNG_BYTES], { type: "image/png" }));
    expect(png.ok).toBe(true);
  });

  it("vacio, tipo no-imagen y magic roto son Err sin lanzar", async () => {
    expect((await parseDemoBlob(new Blob([], { type: "image/jpeg" }))).ok).toBe(false);
    expect(
      (await parseDemoBlob(new Blob(["hola"], { type: "text/plain" }))).ok,
    ).toBe(false);
    expect(
      (await parseDemoBlob(new Blob(["hola"], { type: "image/jpeg" }))).ok,
    ).toBe(false);
    expect((await parseDemoBlob("x.jpg")).ok).toBe(false);
    expect((await parseDemoBlob(null)).ok).toBe(false);
    expect((await parseDemoBlob(undefined)).ok).toBe(false);
    await expect(parseDemoBlob({})).resolves.toBeDefined();
  });
});

describe("demo-loader: loadDemoBlobs railway, nunca lanza", () => {
  it("par 200 con JPEG da Ok con el par", async () => {
    const fakeFetch = (async (): Promise<Response> =>
      demoResponse(jpegBlob())) as unknown as typeof fetch;
    const loaded = await loadDemoBlobs(DEMO_PATHS, { fetchImpl: fakeFetch });
    expect(loaded.ok).toBe(true);
    if (loaded.ok) {
      expect(await demoImageToBlob(loaded.value.a).arrayBuffer()).toEqual(
        await jpegBlob().arrayBuffer(),
      );
      expect(await demoImageToBlob(loaded.value.b).arrayBuffer()).toEqual(
        await jpegBlob().arrayBuffer(),
      );
    }
  });

  it("404 en B es Err BadStatus con slot y codigo", async () => {
    const fakeFetch = (async (input: string): Promise<Response> =>
      String(input).endsWith("obama-b.jpg")
        ? new Response("no", { status: 404 })
        : demoResponse(jpegBlob())) as unknown as typeof fetch;
    const loaded = await loadDemoBlobs(DEMO_PATHS, { fetchImpl: fakeFetch });
    expect(loaded.ok).toBe(false);
    if (!loaded.ok) {
      expect(loaded.error.kind).toBe("DemoLoadFailed");
      expect(loaded.error.which).toBe("b");
      expect(loaded.error.detail).toBe("status");
      expect(loaded.error.status).toBe(404);
    }
  });

  it("red caida es Err network sin lanzar", async () => {
    const fakeFetch = (async (): Promise<Response> => {
      throw new Error("down");
    }) as unknown as typeof fetch;
    const loaded = await loadDemoBlobs(DEMO_PATHS, { fetchImpl: fakeFetch });
    expect(loaded.ok).toBe(false);
    if (!loaded.ok) {
      expect(loaded.error.kind).toBe("DemoLoadFailed");
      expect(loaded.error.detail).toBe("network");
    }
    await expect(
      loadDemoBlobs(DEMO_PATHS, { fetchImpl: fakeFetch }),
    ).resolves.toBeDefined();
  });
});

describe("demo-loader: blobsToFiles listos para FormData", () => {
  it("nombres fijos demo-obama + tipo y bytes preservados", async () => {
    const a = await parseDemoBlob(jpegBlob());
    const b = await parseDemoBlob(jpegBlob());
    expect(a.ok).toBe(true);
    expect(b.ok).toBe(true);
    if (!a.ok || !b.ok) return;
    const files = blobsToFiles(a.value, b.value);
    expect(files.fileA.name).toBe(DEMO_FILE_NAMES.a);
    expect(files.fileB.name).toBe(DEMO_FILE_NAMES.b);
    expect(files.fileA.type).toBe("image/jpeg");
    expect(files.fileA.size).toBeGreaterThan(0);
    expect(await files.fileA.arrayBuffer()).toEqual(await jpegBlob().arrayBuffer());
  });
});
