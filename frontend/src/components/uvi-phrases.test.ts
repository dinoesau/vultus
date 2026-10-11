import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import {
  PHRASES,
  PHRASE_INTERVAL_MS,
  nextPhraseIndex,
  phraseAt,
  startPhraseRotation,
  stopPhraseRotation,
} from "./uvi-phrases.ts";
import type { PhraseClock } from "./uvi-phrases.ts";

// Seam bajo prueba: modulo uvi-phrases.ts (constantes + next/start/stop
// totales con Result, clock inyectable => sin DOM) + snapshot de
// index.astro (cableado start/stop, role=status, mono 13px).
// Sin jsdom: FakeClock captura el intervalo y dispara ticks a mano.

function makeFakeClock(): {
  clock: PhraseClock;
  intervals: number[];
  cleared: unknown[];
  tickAll: (times?: number) => void;
  liveCount: () => number;
} {
  const cbs = new Map<number, () => void>();
  let nextId = 1;
  const intervals: number[] = [];
  const cleared: unknown[] = [];
  const clock: PhraseClock = {
    setInterval: (cb: () => void, ms: number): unknown => {
      const id = nextId;
      nextId += 1;
      cbs.set(id, cb);
      intervals.push(ms);
      return id;
    },
    clearInterval: (id: unknown): void => {
      cleared.push(id);
      if (typeof id === "number") cbs.delete(id);
    },
  };
  return {
    clock,
    intervals,
    cleared,
    tickAll: (times = 1): void => {
      for (let n = 0; n < times; n += 1) {
        for (const cb of [...cbs.values()]) cb();
      }
    },
    liveCount: (): number => cbs.size,
  };
}

function readIndex(): string {
  return readFileSync(new URL("../pages/index.astro", import.meta.url), "utf8");
}

describe("uvi-phrases: AC1 rota en orden cada ~1.8s en loop", () => {
  it("Given Uvi visible, When pasan 1.8s, Then avanza Exploring/Discovering/Unraveling/Pondering/Grokking en loop", () => {
    expect(PHRASES).toEqual([
      "Exploring...",
      "Discovering...",
      "Unraveling...",
      "Pondering...",
      "Grokking...",
    ]);
    expect(PHRASE_INTERVAL_MS).toBe(1800);
    const fake = makeFakeClock();
    const seen: string[] = [];
    const started = startPhraseRotation((p) => seen.push(p), fake.clock);
    expect(started.ok).toBe(true);
    // Render inmediato de la primera + un avance por tick.
    expect(seen).toEqual(["Exploring..."]);
    expect(fake.intervals).toEqual([1800]);
    fake.tickAll(5);
    expect(seen).toEqual([
      "Exploring...",
      "Discovering...",
      "Unraveling...",
      "Pondering...",
      "Grokking...",
      "Exploring...",
    ]);
  });

  it("Given HQ de minutos, When pasan cientos de ticks, Then sigue rotando sin congelarse", () => {
    const fake = makeFakeClock();
    const seen: string[] = [];
    const started = startPhraseRotation((p) => seen.push(p), fake.clock);
    expect(started.ok).toBe(true);
    // ~6 min de HQ a 1.8s = 200 ticks: un solo setInterval, sin deriva.
    fake.tickAll(200);
    expect(fake.intervals).toEqual([1800]);
    expect(seen.length).toBe(201);
    expect(seen[200]).toBe(PHRASES[200 % PHRASES.length]);
  });

  it("next/phraseAt son totales: indice fuera de rango o sucio es Err, nunca lanza", () => {
    expect(nextPhraseIndex(4).ok).toBe(true);
    const wrapped = nextPhraseIndex(4);
    if (wrapped.ok) expect(wrapped.value).toBe(0);
    else expect.unreachable();
    const at = phraseAt(0);
    if (at.ok) expect(at.value).toBe("Exploring...");
    else expect.unreachable();
    expect(phraseAt(-1).ok).toBe(false);
    expect(phraseAt(5).ok).toBe(false);
    expect(phraseAt("x").ok).toBe(false);
    expect(nextPhraseIndex(Number.NaN).ok).toBe(false);
    expect(() => phraseAt("x")).not.toThrow();
    expect(() => nextPhraseIndex(Number.NaN)).not.toThrow();
  });
});

describe("uvi-phrases: AC2 stop oculta y limpia sin fugas", () => {
  it("Given rotacion viva, When stop, Then clearInterval y ningun render mas", () => {
    const fake = makeFakeClock();
    const seen: string[] = [];
    const started = startPhraseRotation((p) => seen.push(p), fake.clock);
    if (!started.ok) expect.unreachable();
    const stopped = stopPhraseRotation(started.value);
    expect(stopped.ok).toBe(true);
    expect(fake.cleared.length).toBe(1);
    expect(fake.liveCount()).toBe(0);
    fake.tickAll(3);
    expect(seen).toEqual(["Exploring..."]);
  });

  it("start con render sucio y stop con handle sucio son Err, nunca lanzan", () => {
    expect(startPhraseRotation(null).ok).toBe(false);
    expect(stopPhraseRotation(null).ok).toBe(false);
    expect(stopPhraseRotation({}).ok).toBe(false);
    expect(() => startPhraseRotation(null)).not.toThrow();
    expect(() => stopPhraseRotation(null)).not.toThrow();
  });

  it("index cablea start en uviShow y stop+clear en uviHide (reusa hide existente)", () => {
    const src = readIndex();
    expect(src.includes('from "../components/uvi-phrases.ts"')).toBe(true);
    // uviShow arranca rotacion; uviHide la detiene y limpia el texto.
    expect(src.includes("startPhraseRotation")).toBe(true);
    expect(src.includes("stopPhraseRotation")).toBe(true);
    // Sin setTimeout recursivo para frases: el unico timer vive en el modulo
    // con un solo setInterval (sin deriva en HQ largo).
    expect(src.includes("setTimeout(() => void fetchResult")).toBe(true);
    const mod = readFileSync(
      new URL("./uvi-phrases.ts", import.meta.url),
      "utf8",
    );
    expect(mod.split("setInterval").length - 1).toBeGreaterThanOrEqual(2);
    expect(mod.includes("setTimeout")).toBe(false);
    expect(mod.includes("PHRASE_INTERVAL_MS")).toBe(true);
  });
});

describe("uvi-phrases: AC3 live region unica mono 13px", () => {
  it("#uvi-text es role=status aria-live=polite, mono 13px, sin animacion extra", () => {
    const src = readIndex();
    const pAt = src.indexOf('id="uvi-text"');
    expect(pAt).toBeGreaterThan(-1);
    const tag = src.slice(pAt - 160, pAt + 60);
    expect(tag.includes('role="status"')).toBe(true);
    expect(tag.includes('aria-live="polite"')).toBe(true);
    // Un solo nodo live en el bloque Uvi: no spamea el lector.
    const uviAt = src.indexOf('id="uvi"');
    const tail = src.slice(uviAt, uviAt + 7000);
    expect(tail.split("aria-live").length - 1).toBe(1);
    // Mono 13px segun story (antes 12px sistema).
    expect(src.includes("#uvi-text")).toBe(true);
    expect(src.includes("13px")).toBe(true);
    expect(src.includes("JetBrains Mono")).toBe(true);
  });
});
