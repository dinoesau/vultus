import { test, expect, type Page } from "@playwright/test";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";

// S6 (epica E2 #172): el canvas del boot real ya toco 2D via renderNeutral,
// asi que despachar vultus:meshes con GLBs reales debe llegar a ready solo
// con el fix S4 (canvas WebGL dedicado). Sin el fix, el WebGLRenderer no se
// puede adjuntar ("Canvas has an existing context of a different type") y el
// estado se queda en invalido: este test falla en rojo hasta que S4 aterrice.
// Corre sin GPU (SwiftShader software) y sin red (solo localhost + blob:).

const FRONT = process.env.FRONT_URL ?? "http://localhost:4321";
const FIXTURES = fileURLToPath(new URL("./fixtures/", import.meta.url));

// Pin estructural del golden (fuente independiente: issue #179, no el codigo):
// 2 meshes (A+B), 10046 verts (2x5023), materiales SkinPBR + EyePBR, PBR 512x512.
const EXPECTED_TOTAL_VERTS = 10046;
const EXPECTED_MATERIALS = ["SkinPBR", "EyePBR"] as const;
const EXPECTED_PBR_SIZE = 512;

test.use({
  launchOptions: {
    args: [
      "--use-gl=angle",
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--disable-dev-shm-usage",
    ],
  },
});

interface GlbSummary {
  readonly meshCount: number;
  readonly primCount: number;
  readonly positionVerts: number;
  readonly materialNames: readonly string[];
  readonly imageCount: number;
}

// Parse de borde del GLB en Node (sin WebGL): valida magic + total y resume
// meshes/verts/materiales desde el chunk JSON. Sin casts fuera de este parser.
function summarizeGlb(raw: Buffer): GlbSummary {
  const magic = raw.subarray(0, 4).toString("ascii");
  if (magic !== "glTF") throw new Error(`golden sin magia glTF (llego ${magic})`);
  const total = raw.readUInt32LE(8);
  if (total !== raw.length) throw new Error(`golden con total descuadrado (${total} vs ${raw.length})`);
  const jsonLen = raw.readUInt32LE(12);
  const doc = JSON.parse(raw.subarray(20, 20 + jsonLen).toString("utf-8")) as {
    readonly meshes?: ReadonlyArray<{ readonly primitives?: ReadonlyArray<unknown> }>;
    readonly materials?: ReadonlyArray<{ readonly name?: unknown }>;
    readonly accessors?: ReadonlyArray<{ readonly type?: unknown; readonly componentType?: unknown; readonly count?: unknown }>;
    readonly images?: ReadonlyArray<unknown>;
  };
  let primCount = 0;
  // Las 2 primitivas (piel + ojo) comparten el mismo anillo POSITION del
  // builder: se desduplica por indice de accessor para contar verts unicos.
  const seenPosition = new Set<number>();
  let positionVerts = 0;
  const meshes = doc.meshes ?? [];
  const accessors = doc.accessors ?? [];
  for (const mesh of meshes) {
    for (const prim of mesh.primitives ?? []) {
      primCount += 1;
      const attrs = (prim as { readonly attributes?: { readonly POSITION?: unknown } }).attributes;
      const idx = attrs?.POSITION;
      if (typeof idx !== "number") throw new Error("primitiva sin POSITION");
      const accessor = accessors[idx];
      if (accessor?.type !== "VEC3" || typeof accessor.count !== "number") {
        throw new Error("accessor POSITION inesperado");
      }
      if (!seenPosition.has(idx)) {
        seenPosition.add(idx);
        positionVerts += accessor.count;
      }
    }
  }
  const materialNames = (doc.materials ?? []).map((m) =>
    typeof m.name === "string" ? m.name : "",
  );
  return {
    meshCount: meshes.length,
    primCount,
    positionVerts,
    materialNames,
    imageCount: doc.images?.length ?? 0,
  };
}

function readFixture(name: string): Buffer {
  return readFileSync(`${FIXTURES}${name}`);
}

// Given compartido: boot real con neutro 2D corriendo (path que la repro E1
// omitia). El poll sincroniza con la isla: si el canvas ya tiene 2D, el
// listener de vultus:meshes ya esta adjunto y el dispatch no se pierde en frio.
async function expectNeutralBoot(page: Page): Promise<void> {
  await expect
    .poll(
      async () =>
        page.evaluate(() => {
          const canvas = document.querySelector('[data-testid="viewer-3d-a"]');
          if (!(canvas instanceof HTMLCanvasElement)) return "sin-canvas";
          return canvas.getContext("2d") === null ? "sin-2d" : "con-2d";
        }),
      { timeout: 30_000 },
    )
    .toBe("con-2d");
}

interface MeshPayload {
  readonly meshA: Buffer;
  readonly meshB: Buffer;
  readonly pbrA: Buffer;
  readonly pbrB: Buffer;
}

// When compartido: despacha vultus:meshes con Blobs reales por blob:. Parse
// de borde en Node (base64) y en pagina (atob), sin casts fuera de este borde.
async function dispatchMeshes(page: Page, files: MeshPayload): Promise<void> {
  const payload = {
    meshA: files.meshA.toString("base64"),
    meshB: files.meshB.toString("base64"),
    pbrA: files.pbrA.toString("base64"),
    pbrB: files.pbrB.toString("base64"),
  };
  await page.evaluate(
    (p: { meshA: string; meshB: string; pbrA: string; pbrB: string }) => {
      const toBlob = (b64: string, type: string): Blob => {
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
        return new Blob([bytes.buffer as ArrayBuffer], { type });
      };
      window.dispatchEvent(
        new CustomEvent("vultus:meshes", {
          detail: {
            meshA: toBlob(p.meshA, "model/gltf-binary"),
            meshB: toBlob(p.meshB, "model/gltf-binary"),
            pbrA: toBlob(p.pbrA, "image/png"),
            pbrB: toBlob(p.pbrB, "image/png"),
          },
        }),
      );
    },
    payload,
  );
}

function pngSize(raw: Buffer): { readonly width: number; readonly height: number } {
  if (raw.subarray(0, 8).toString("binary") !== "\x89PNG\r\n\x1a\n") {
    throw new Error("overlay PBR sin magia PNG");
  }
  return { width: raw.readUInt32BE(16), height: raw.readUInt32BE(20) };
}

test("golden GLB pin estructural: 2 meshes, 10046 verts, SkinPBR+EyePBR, PBR 512x512", async () => {
  const summaries = [summarizeGlb(readFixture("mesh_a.glb")), summarizeGlb(readFixture("mesh_b.glb"))];
  for (const summary of summaries) {
    expect(summary.meshCount).toBe(1);
    expect(summary.primCount).toBe(2);
    expect(summary.positionVerts).toBe(EXPECTED_TOTAL_VERTS / 2);
    expect(summary.materialNames).toEqual([...EXPECTED_MATERIALS]);
    expect(summary.imageCount).toBe(2);
  }
  // 2 meshes x 5023 verts unicos = 10046 (issue #179).
  const combined = summaries.reduce((n, s) => n + s.positionVerts, 0);
  expect(combined).toBe(EXPECTED_TOTAL_VERTS);
  for (const name of ["pbr_a.png", "pbr_b.png"] as const) {
    const size = pngSize(readFixture(name));
    expect(size.width).toBe(EXPECTED_PBR_SIZE);
    expect(size.height).toBe(EXPECTED_PBR_SIZE);
  }
});

test("boot real 2D + golden llega a ready (rojo sin fix S4)", async ({ page }) => {
  test.setTimeout(120_000);
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => pageErrors.push(String(err)));
  // Sin red saliente: solo el servidor local y blob:/data: del propio test.
  await page.route(/^https?:\/\/(?!localhost|127\.0\.0\.1).*/, (route) => route.abort());

  await page.goto(FRONT);
  // Prueba directa del modo de fallo: Chrome dispara webglcontextcreationerror
  // con "Canvas has an existing context of a different type" cuando el 2D ya
  // clavo el canvas (dictamen E1 #171). Se registra en la ventana para
  // volcarlo como evidencia en el finally, sin acoplarse al log de three.
  await page.evaluate(() => {
    const store: string[] = [];
    (window as unknown as { __s6ctxErrors?: string[] }).__s6ctxErrors = store;
    for (const testId of ["viewer-3d-a", "viewer-3d-b"]) {
      const canvas = document.querySelector(`[data-testid="${testId}"]`);
      if (canvas instanceof HTMLCanvasElement) {
        canvas.addEventListener("webglcontextcreationerror", (ev) => {
          store.push(`${testId}: ${(ev as WebGLContextEvent).statusMessage ?? "?"}`);
        });
      }
    }
  });
  const status = page.locator("#viewer-3d-status");
  await expectNeutralBoot(page);

  // When: llega vultus:meshes con el golden (GLB + PBR reales por blob:).
  await dispatchMeshes(page, {
    meshA: readFixture("mesh_a.glb"),
    meshB: readFixture("mesh_b.glb"),
    pbrA: readFixture("pbr_a.png"),
    pbrB: readFixture("pbr_b.png"),
  });

  // Then: con el fix S4 llega a ready; sin el fix se queda en invalido porque
  // el canvas ya tiene contexto 2D ("different type"). La evidencia del rojo
  // queda en consoleErrors/pageErrors del log de este test (finally: se
  // imprime tambien cuando la asercion falla).
  let reachedReady = false;
  try {
    await expect(status).toContainText(/cara real lista/, { timeout: 60_000 });
    reachedReady = true;
  } finally {
    const ctxErrors = await page
      .evaluate(() => (window as unknown as { __s6ctxErrors?: unknown }).__s6ctxErrors)
      .then((raw): readonly string[] => {
        if (!Array.isArray(raw)) return [];
        return raw.filter((e): e is string => typeof e === "string");
      })
      .catch(() => [] as readonly string[]);
    // eslint-disable-next-line no-console
    console.log(
      `[s6] reachedReady=${reachedReady} ctxCreationErrors=${ctxErrors.length} consoleErrors=${consoleErrors.length} pageErrors=${pageErrors.length}`,
    );
    for (const line of [...ctxErrors, ...consoleErrors, ...pageErrors].slice(0, 8)) {
      // eslint-disable-next-line no-console
      console.log(`[s6-failure-evidence] ${line.slice(0, 300)}`);
    }
  }
  expect(reachedReady).toBe(true);
  for (const id of ["download-mesh-a", "download-mesh-b"] as const) {
    const link = page.locator(`#${id}`);
    await expect(link).toBeVisible();
    await expect(link).toHaveAttribute("href", /^blob:/);
  }
  // Los canvas dejan el fondo neutro: capturas no vacias y distintas.
  const shotA = await page.getByTestId("viewer-3d-a").screenshot();
  const shotB = await page.getByTestId("viewer-3d-b").screenshot();
  expect(shotA.length).toBeGreaterThan(1500);
  expect(shotB.length).toBeGreaterThan(1500);
  expect(shotA.equals(shotB)).toBe(false);
});

test("fixture corrupto falla en fase parse, nunca en ready", async ({ page }) => {
  test.setTimeout(120_000);
  await page.route(/^https?:\/\/(?!localhost|127\.0\.0\.1).*/, (route) => route.abort());
  await page.goto(FRONT);
  const status = page.locator("#viewer-3d-status");
  await expectNeutralBoot(page);

  const good = readFixture("mesh_a.glb");
  const badMagic = Buffer.from(good);
  badMagic.write("XXXX", 0);
  const badTotal = Buffer.from(good);
  badTotal.writeUInt32LE(good.length + 999, 8);
  const pbr = readFixture("pbr_a.png");

  for (const [label, broken] of [["magia-rota", badMagic], ["total-descuadrado", badTotal]] as const) {
    await dispatchMeshes(page, { meshA: broken, meshB: broken, pbrA: pbr, pbrB: pbr });
    // Fase parse segun S5: mensaje de GLB invalido/parse con causa, nunca
    // WebGL ni ready. El regex acepta el mensaje actual y el fraseo por fases
    // de S5 (el vocabulario cerrado vive en three-rig.ts).
    await expect(status).toContainText(/parse|GLB invalido/i, { timeout: 30_000 });
    await expect(status).not.toContainText(/cara real lista/);
    void label;
  }
});
