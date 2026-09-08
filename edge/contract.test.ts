import { describe, expect, it } from "vitest";
import {
  MAX_IMAGE_BYTES,
  PROGRESS_BAKE,
  PROGRESS_DONE,
  PROGRESS_FLAME,
  PROGRESS_FREEUV,
  PROGRESS_LANDMARKS,
  RESULT_TTL_SECONDS,
  STAGES,
  STATUSES,
  TERMINAL_STATUSES,
  hasSupportedMagic,
  isJobStatus,
  isUuid,
  isValidProgress,
  isValidStage,
  jobIdToString,
  parseJobId,
  parseJobStatus,
  parseProgress,
  parseStage,
  parseTtlSecs,
  parseTtlSecsBranded,
  progressToNumber,
  ttlToNumber,
} from "./contract";

describe("contrato edge como fuente de verdad", () => {
  it("parsea uuid con trim y rechaza roto sin lanzar", () => {
    const ok = parseJobId("  11111111-1111-4111-8111-111111111111  ");
    expect(ok.ok).toBe(true);
    if (ok.ok) expect(jobIdToString(ok.value)).toBe("11111111-1111-4111-8111-111111111111");
    expect(parseJobId("not-a-uuid").ok).toBe(false);
    expect(parseJobId("").ok).toBe(false);
    expect(parseJobId(123).ok).toBe(false);
    expect(isUuid("11111111-1111-4111-8111-111111111111")).toBe(true);
  });

  it("ttl clamp 1..3600 default 60 nunca NaN", () => {
    expect(RESULT_TTL_SECONDS).toBe(60);
    expect(parseTtlSecs(null)).toBe(60);
    expect(parseTtlSecs(undefined)).toBe(60);
    expect(parseTtlSecs("nope")).toBe(60);
    expect(parseTtlSecs("0")).toBe(1);
    expect(parseTtlSecs("9999")).toBe(3600);
    expect(parseTtlSecs("60")).toBe(60);
    expect(ttlToNumber(parseTtlSecsBranded("60"))).toBe(60);
    expect(ttlToNumber(parseTtlSecsBranded("0"))).toBe(1);
  });

  it("progress solo 0..1 finito con Result", () => {
    const ok = parseProgress(0.4);
    expect(ok.ok).toBe(true);
    if (ok.ok) expect(progressToNumber(ok.value)).toBeCloseTo(0.4);
    expect(parseProgress(-0.1).ok).toBe(false);
    expect(parseProgress(1.1).ok).toBe(false);
    expect(parseProgress(Number.NaN).ok).toBe(false);
    expect(parseProgress("0.5").ok).toBe(false);
    expect(isValidProgress(0.4)).toBe(true);
  });

  it("stage con Result y orden canonico", () => {
    expect([...STAGES]).toEqual(["queued", "landmarks", "flame", "freeuv", "bake", "done"]);
    const ok = parseStage("flame");
    expect(ok.ok).toBe(true);
    if (ok.ok) expect(ok.value).toBe("flame");
    expect(parseStage("nope").ok).toBe(false);
    expect(isValidStage("bake")).toBe(true);
  });

  it("magics JPEG PNG y limite 8MB", () => {
    expect(MAX_IMAGE_BYTES).toBe(8 * 1024 * 1024);
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00]);
    const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0x00]);
    expect(hasSupportedMagic(png)).toBe(true);
    expect(hasSupportedMagic(jpeg)).toBe(true);
    expect(hasSupportedMagic(new Uint8Array([1, 2, 3]))).toBe(false);
  });

  it("status union con orden canonico y Result", () => {
    expect([...STATUSES]).toEqual(["queued", "processing", "done", "failed", "expired"]);
    expect([...TERMINAL_STATUSES]).toEqual(["done", "failed", "expired"]);
    const ok = parseJobStatus("processing");
    expect(ok.ok).toBe(true);
    if (ok.ok) expect(ok.value).toBe("processing");
    expect(parseJobStatus("nope").ok).toBe(false);
    expect(parseJobStatus("").ok).toBe(false);
    expect(parseJobStatus(42).ok).toBe(false);
    expect(isJobStatus("done")).toBe(true);
    expect(isJobStatus("queued")).toBe(true);
    expect(isJobStatus("bogus")).toBe(false);
  });

  it("hitos de progreso espejan pipeline 0.15/0.40/0.75/0.95/1.0", () => {
    expect(PROGRESS_LANDMARKS).toBeCloseTo(0.15);
    expect(PROGRESS_FLAME).toBeCloseTo(0.4);
    expect(PROGRESS_FREEUV).toBeCloseTo(0.75);
    expect(PROGRESS_BAKE).toBeCloseTo(0.95);
    expect(PROGRESS_DONE).toBe(1.0);
  });
});
