import { describe, expect, it } from "vitest";
import {
  HEALTH_POLL_INTERVAL_MS,
  HQ_POLL_INTERVAL_MS,
  HQ_RESULT_RETRY,
  POLL_INTERVAL_MS,
  QUALITY_HQ,
  RESULT_RETRY,
  healthDotState,
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

// Story-152 pill Health: un test por criterio Given-When-Then.
describe("compare-client health pill: dot vivo desde GET /health", () => {
  it("AC1 API OK con contract v3: dot verde con tooltip contract v3", () => {
    const state = healthDotState({ contract_version: 3 }, true);
    expect(state).toEqual({ ok: true, title: "contract v3" });
  });

  it("AC2 API caida o mismatch: dot rojo (5xx, JSON roto, version distinta)", () => {
    expect(healthDotState({ contract_version: 3 }, false).ok).toBe(false);
    expect(healthDotState(null, true)).toEqual({ ok: false, title: "down" });
    const mismatch = healthDotState({ contract_version: 2 }, true);
    expect(mismatch.ok).toBe(false);
    expect(mismatch.title).toContain("v2");
    expect(healthDotState({ contract_version: 3 }, false).title).toBe("down");
  });

  it("AC3 cualquier estado: tooltip con detalle no vacio, sin throw", () => {
    expect(healthDotState({ contract_version: 3 }, true).title.length).toBeGreaterThan(0);
    expect(healthDotState(null, true).title.length).toBeGreaterThan(0);
    expect(healthDotState({ contract_version: 9 }, true).title.length).toBeGreaterThan(0);
    expect(healthDotState(undefined, false).title.length).toBeGreaterThan(0);
  });

  it("AC4 intervalo de refresco 30s sin magicos en pages", () => {
    expect(HEALTH_POLL_INTERVAL_MS).toBe(30_000);
  });
});
