import { describe, expect, it } from "vitest";
import {
  DEMO_LOAD_ERROR_MESSAGE,
  DEMO_PATHS,
  blobsToFiles,
  demoFileName,
  demoImageToBlob,
  demoLoadErrorMessage,
  loadDemoBlobs,
  parseDemoBlob,
  sniffDemoMime,
  type DemoImage,
} from "./demo-loader.ts";

const JPEG_BYTES = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
const PNG_BYTES = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function jpegBlob(type = "image/jpeg"): Blob {
  return new Blob([JPEG_BYTES], { type });
}

function demoResponse(blob: Blob, status = 200): Response {
  return new Response(blob, {
    status,
    headers: { "content-type": blob.type || "application/octet-stream" },
  });
}

async function mustParse(blob: Blob): Promise<DemoImage> {
  const parsed = await parseDemoBlob(blob);
  expect(parsed.ok).toBe(true);
  if (!parsed.ok) throw new Error("unreachable in test");
  return parsed.value;
}

describe("demo-loader: sniffDemoMime por contenido, no por type", () => {
  it("JPEG y PNG reales dan su MIME aunque type mienta", async () => {
    expect(await sniffDemoMime(jpegBlob())).toBe("image/jpeg");
    expect(await sniffDemoMime(jpegBlob(""))).toBe("image/jpeg");
    expect(await sniffDemoMime(jpegBlob("image/jpg"))).toBe("image/jpeg");
    expect(await sniffDemoMime(jpegBlob("application/octet-stream"))).toBe("image/jpeg");
    expect(
      await sniffDemoMime(new Blob([PNG_BYTES], { type: "" })),
    ).toBe("image/png");
  });

  it("basura da null aunque declare image/jpeg", async () => {
    expect(await sniffDemoMime(new Blob(["hola"], { type: "image/jpeg" }))).toBe(null);
    expect(await sniffDemoMime(new Blob([], { type: "image/jpeg" }))).toBe(null);
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

  it("type vacio o alias con magic valido dan Ok (el host puede mentir)", async () => {
    expect((await parseDemoBlob(jpegBlob(""))).ok).toBe(true);
    expect((await parseDemoBlob(jpegBlob("image/jpg"))).ok).toBe(true);
    expect((await parseDemoBlob(jpegBlob("image/pjpeg"))).ok).toBe(true);
    expect(
      (await parseDemoBlob(new Blob([PNG_BYTES], { type: "" }))).ok,
    ).toBe(true);
  });

  it("vacio, no-blob y magic roto son Err sin lanzar", async () => {
    const empty = await parseDemoBlob(new Blob([], { type: "image/jpeg" }));
    expect(empty.ok).toBe(false);
    if (!empty.ok) expect(empty.error.reason).toBe("empty");
    const garbage = await parseDemoBlob(new Blob(["hola"], { type: "image/jpeg" }));
    expect(garbage.ok).toBe(false);
    if (!garbage.ok) expect(garbage.error.reason).toBe("magic");
    const text = await parseDemoBlob(new Blob(["hola"], { type: "text/plain" }));
    expect(text.ok).toBe(false);
    expect((await parseDemoBlob("x.jpg")).ok).toBe(false);
    expect((await parseDemoBlob(null)).ok).toBe(false);
    expect((await parseDemoBlob(undefined)).ok).toBe(false);
    await expect(parseDemoBlob({})).resolves.toBeDefined();
  });

  it("brand: un Blob crudo no es DemoImage (falla en tsc)", () => {
    // @ts-expect-error DemoImage exige el brand de parseDemoBlob
    const notAnImage: DemoImage = new Blob([JPEG_BYTES], { type: "image/jpeg" });
    expect(notAnImage).toBeDefined();
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

  it("404 en B es Err status con slot y codigo", async () => {
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

  it("doble fallo: gana el slot a (prioridad documentada)", async () => {
    const fakeFetch = (async (input: string): Promise<Response> =>
      String(input).endsWith("obama-a.jpg")
        ? new Response("no", { status: 404 })
        : demoResponse(new Blob(["hola"], { type: "image/jpeg" }))) as unknown as typeof fetch;
    const loaded = await loadDemoBlobs(DEMO_PATHS, { fetchImpl: fakeFetch });
    expect(loaded.ok).toBe(false);
    if (!loaded.ok) {
      expect(loaded.error.which).toBe("a");
      expect(loaded.error.detail).toBe("status");
    }
  });

  it("payload corrupto es blob-invalid, distinto de fallo de transporte", async () => {
    const fakeFetch = (async (): Promise<Response> =>
      demoResponse(
        new Blob(["hola"], { type: "image/jpeg" }),
      )) as unknown as typeof fetch;
    const loaded = await loadDemoBlobs(DEMO_PATHS, { fetchImpl: fakeFetch });
    expect(loaded.ok).toBe(false);
    if (!loaded.ok) expect(loaded.error.detail).toBe("blob-invalid");
  });

  it("red caida y abort son Err network sin lanzar", async () => {
    const downFetch = (async (): Promise<Response> => {
      throw new Error("down");
    }) as unknown as typeof fetch;
    const down = await loadDemoBlobs(DEMO_PATHS, { fetchImpl: downFetch });
    expect(down.ok).toBe(false);
    if (!down.ok) {
      expect(down.error.kind).toBe("DemoLoadFailed");
      expect(down.error.detail).toBe("network");
    }
    const ctrl = new AbortController();
    ctrl.abort();
    const aborted = await loadDemoBlobs(DEMO_PATHS, { signal: ctrl.signal });
    expect(aborted.ok).toBe(false);
    if (!aborted.ok) expect(aborted.error.detail).toBe("network");
    await expect(
      loadDemoBlobs(DEMO_PATHS, { fetchImpl: downFetch }),
    ).resolves.toBeDefined();
  });
});

describe("demo-loader: blobsToFiles listos para FormData", () => {  it("JPEG da .jpg con tipo y bytes preservados", async () => {
    const files = await blobsToFiles(await mustParse(jpegBlob()), await mustParse(jpegBlob()));
    expect(files.ok).toBe(true);
    if (!files.ok) return;
    expect(files.value.fileA.name).toBe("demo-obama-a.jpg");
    expect(files.value.fileB.name).toBe("demo-obama-b.jpg");
    expect(files.value.fileA.type).toBe("image/jpeg");
    expect(files.value.fileA.size).toBeGreaterThan(0);
    expect(await files.value.fileA.arrayBuffer()).toEqual(
      await jpegBlob().arrayBuffer(),
    );
  });

  it("PNG da .png con su tipo (extension y bytes coinciden)", async () => {
    const png = await mustParse(new Blob([PNG_BYTES], { type: "" }));
    const files = await blobsToFiles(png, png);
    expect(files.ok).toBe(true);
    if (!files.ok) return;
    expect(files.value.fileA.name).toBe("demo-obama-a.png");
    expect(demoFileName("b", "image/png")).toBe("demo-obama-b.png");
    expect(files.value.fileA.type).toBe("image/png");
    expect(await files.value.fileA.arrayBuffer()).toEqual(
      await new Blob([PNG_BYTES]).arrayBuffer(),
    );
  });
});

describe("demo-loader: demoLoadErrorMessage literal único (#123)", () => {
  it("el literal es el exigible por el issue", () => {
    expect(DEMO_LOAD_ERROR_MESSAGE).toBe("Could not load demo. Try again");
  });

  it("todo detail de carga mapea al mismo literal", () => {
    expect(
      demoLoadErrorMessage({ kind: "DemoLoadFailed", which: "a", detail: "network" }),
    ).toBe(DEMO_LOAD_ERROR_MESSAGE);
    expect(
      demoLoadErrorMessage({ kind: "DemoLoadFailed", which: "b", detail: "status", status: 404 }),
    ).toBe(DEMO_LOAD_ERROR_MESSAGE);
    expect(
      demoLoadErrorMessage({ kind: "DemoLoadFailed", which: "a", detail: "blob-transport" }),
    ).toBe(DEMO_LOAD_ERROR_MESSAGE);
    expect(
      demoLoadErrorMessage({ kind: "DemoLoadFailed", which: "b", detail: "blob-invalid" }),
    ).toBe(DEMO_LOAD_ERROR_MESSAGE);
  });

  it("blob inválido y DOM mapean al mismo literal", () => {
    expect(
      demoLoadErrorMessage({ kind: "InvalidDemoBlob", reason: "magic" }),
    ).toBe(DEMO_LOAD_ERROR_MESSAGE);
    expect(demoLoadErrorMessage({ kind: "DemoDomError" })).toBe(DEMO_LOAD_ERROR_MESSAGE);
  });
});
