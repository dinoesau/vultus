import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

// P2-10: lo que README.md afirma (magic, 1024x1024, presupuesto de bytes)
// se verifica aquí. Sin este test un recompress/wrong-crop pasaría E1
// (naturalWidth>0 vale para cualquier imagen) mientras el README miente.
// Sin deps nuevas: el SOF JPEG se parsea a mano (scan de markers).

const MAX_IMAGE_BYTES = 8 * 1024 * 1024; // espejo backend/domain.py:37

function jpegDimensions(bytes: Uint8Array): { w: number; h: number } | null {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return null;
  let i = 2;
  while (i + 3 < bytes.length) {
    if (bytes[i] !== 0xff) return null;
    const marker = bytes[i + 1];
    // SOF0-SOF15 salvo C4/C8/CC: traen alto/ancho big-endian tras 5 bytes.
    if (
      marker >= 0xc0 &&
      marker <= 0xcf &&
      marker !== 0xc4 &&
      marker !== 0xc8 &&
      marker !== 0xcc
    ) {
      return { h: (bytes[i + 5] << 8) | bytes[i + 6], w: (bytes[i + 7] << 8) | bytes[i + 8] };
    }
    if (marker === 0xda) return null; // SOS: headers terminados sin SOF.
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      i += 2;
      continue;
    }
    const len = (bytes[i + 2] << 8) | bytes[i + 3];
    if (len < 2) return null;
    i += 2 + len;
  }
  return null;
}

describe("demo-assets: lo que README.md afirma se cumple", () => {
  for (const name of ["obama-a.jpg", "obama-b.jpg"] as const) {
    it(`${name} es JPEG 1024x1024 dentro del presupuesto`, () => {
      const raw = readFileSync(new URL(`../../public/demo/${name}`, import.meta.url));
      const bytes = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
      expect([bytes[0], bytes[1], bytes[2]]).toEqual([0xff, 0xd8, 0xff]);
      expect(jpegDimensions(bytes)).toEqual({ w: 1024, h: 1024 });
      expect(raw.byteLength).toBeGreaterThan(0);
      expect(raw.byteLength).toBeLessThanOrEqual(MAX_IMAGE_BYTES);
    });
  }
});
