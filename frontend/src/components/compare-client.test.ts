import { describe, expect, it } from "vitest";
import {
  HQ_POLL_INTERVAL_MS,
  HQ_RESULT_RETRY,
  POLL_INTERVAL_MS,
  QUALITY_HQ,
  RESULT_RETRY,
  resultTtlNote,
} from "./compare-client";

describe("compare-client HQ: ventanas y calidad sin magicos en pages", () => {
  it("retry standard cabe en TTL 60s, HQ en minutos", () => {
    expect(RESULT_RETRY).toEqual({ maxAttempts: 10, intervalMs: 1000 });
    expect(RESULT_RETRY.maxAttempts * RESULT_RETRY.intervalMs).toBeLessThanOrEqual(60_000);
    expect(HQ_RESULT_RETRY.maxAttempts).toBeGreaterThan(RESULT_RETRY.maxAttempts);
    expect(HQ_RESULT_RETRY.intervalMs).toBeGreaterThanOrEqual(RESULT_RETRY.intervalMs);
    expect(HQ_RESULT_RETRY.maxAttempts * HQ_RESULT_RETRY.intervalMs).toBeGreaterThan(60_000);
    expect(Number.isFinite(HQ_RESULT_RETRY.maxAttempts * HQ_RESULT_RETRY.intervalMs)).toBe(true);
  });

  it("polling HQ mas lento que standard", () => {
    expect(POLL_INTERVAL_MS).toBe(1000);
    expect(HQ_POLL_INTERVAL_MS).toBeGreaterThan(POLL_INTERVAL_MS);
  });

  it("nota TTL distingue ventanas sin exponer ttl crudo", () => {
    expect(QUALITY_HQ).toBe("hq");
    expect(resultTtlNote("abc", false)).toContain("60 s");
    expect(resultTtlNote("abc", true)).toContain("30 min");
    expect(resultTtlNote("abc", true)).toContain("result-abc.zip");
  });
});
