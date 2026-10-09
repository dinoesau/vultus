import { describe, expect, it } from "vitest";
import {
  CONTRACT_VERSION,
  V3_CONTRACT_VERSION_NUMBER,
  V3_FIGURE_ZIP,
  V3_RETENTION_DAYS,
  V3_TRI_COUNT,
  V3_UV_SIZE,
  V3_VERT_COUNT,
  V3_ZIP_MANIFEST,
  V3_ZIP_NAMES,
  ZIP_NAMES,
  contractVersionToNumber,
  parseJobId,
  v3FigurePath,
} from "./contract";

describe("track v3 FFHQ-UV figure (paralelo, forense congelado)", () => {
  it("forense intacto: zip-8 y CONTRACT_VERSION 3", () => {
    expect(contractVersionToNumber(CONTRACT_VERSION)).toBe(3);
    expect(ZIP_NAMES.length).toBe(8);
  });

  it("v3 en paralelo: version 3, malla densa, atlas 1024, bundle disjunto", () => {
    expect(V3_CONTRACT_VERSION_NUMBER).toBe(3);
    expect(V3_VERT_COUNT).toBe(20481);
    expect(V3_TRI_COUNT).toBe(40832);
    expect(V3_UV_SIZE).toBe(1024);
    expect(V3_ZIP_NAMES.length).toBe(6);
    expect(new Set(V3_ZIP_NAMES).size).toBe(6);
    expect(V3_ZIP_MANIFEST.meshDense).toBe("mesh_dense.glb");
    expect(V3_ZIP_MANIFEST.albedo).toBe("albedo_1024.png");
    for (const n of V3_ZIP_NAMES) {
      expect((ZIP_NAMES as readonly string[])).not.toContain(n);
    }
    expect(V3_RETENTION_DAYS).toBeGreaterThanOrEqual(1);
  });

  it("figura v3: zip propio y path tipado disjunto de /v1", () => {
    expect(V3_FIGURE_ZIP).toBe("figure-v3.zip");
    expect(V3_FIGURE_ZIP).not.toBe("result.zip");
    const parsed = parseJobId("11111111-1111-4111-8111-111111111111");
    expect(parsed.ok).toBe(true);
    if (parsed.ok) {
      expect(v3FigurePath(parsed.value)).toBe(
        "/v3/jobs/11111111-1111-4111-8111-111111111111/figure",
      );
    }
  });
});
