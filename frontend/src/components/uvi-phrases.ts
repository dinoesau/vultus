import type { Result } from "./compare-client.ts";

// Seam frases de Uvi (Story-157): rotación ambient bajo el tamagotchi.
// Sin DOM: el core es índice puro + clock inyectable; index.astro aporta el
// render hacia #uvi-text. Fuente única anti Magic Strings: el orden exacto
// y el intervalo viven aquí; el intervalo es UN setInterval (sin timer
// recursivo) para que el HQ de minutos no derive ni se congele.
// Sin `as`: los guards son predicados con typeof/in y lo ajeno (clock,
// render) va por try/catch mapeado a Err. Todo total, nunca lanza.
export const PHRASES = [
  "Exploring...",
  "Discovering...",
  "Unraveling...",
  "Pondering...",
  "Grokking...",
] as const;

export type Phrase = (typeof PHRASES)[number];

export const PHRASE_INTERVAL_MS = 1800;

export type PhraseError =
  | { readonly kind: "BadIndex" }
  | { readonly kind: "BadRender" }
  | { readonly kind: "BadClock" }
  | { readonly kind: "BadHandle" };

export type PhraseRender = (phrase: Phrase) => void;

export type PhraseClock = {
  readonly setInterval: (cb: () => void, ms: number) => unknown;
  readonly clearInterval: (id: unknown) => void;
};

export type PhraseHandle = {
  readonly timerId: unknown;
  readonly clock: PhraseClock;
};

function isPhraseIndex(raw: unknown): raw is number {
  return (
    typeof raw === "number" &&
    Number.isInteger(raw) &&
    raw >= 0 &&
    raw < PHRASES.length
  );
}

function isPhraseRender(raw: unknown): raw is PhraseRender {
  return typeof raw === "function";
}

function isPhraseHandle(raw: unknown): raw is PhraseHandle {
  return (
    typeof raw === "object" &&
    raw !== null &&
    "timerId" in raw &&
    "clock" in raw
  );
}

/**
 * Clock real del navegador. Solo se usa en producción (index.astro no pasa
 * clock); los tests inyectan FakeClock. El id es number en navegador.
 */
export function defaultPhraseClock(): PhraseClock {
  return {
    setInterval: (cb: () => void, ms: number): unknown =>
      globalThis.setInterval(cb, ms),
    clearInterval: (id: unknown): void => {
      if (typeof id === "number") globalThis.clearInterval(id);
    },
  };
}

/** Parse-once del índice. Fuera de rango o sucio va por Err, nunca throw. */
export function phraseAt(rawIndex: unknown): Result<Phrase, PhraseError> {
  if (!isPhraseIndex(rawIndex))
    return { ok: false, error: { kind: "BadIndex" } };
  return { ok: true, value: PHRASES[rawIndex] };
}

/** Siguiente en loop con wrap a 0. Total: el parse clasifica, el avance es módulo. */
export function nextPhraseIndex(
  rawCurrent: unknown,
): Result<number, PhraseError> {
  if (!isPhraseIndex(rawCurrent))
    return { ok: false, error: { kind: "BadIndex" } };
  return { ok: true, value: (rawCurrent + 1) % PHRASES.length };
}

/**
 * Arranca la rotación: pinta la primera frase ya y avanza una por intervalo.
 * El índice vive en el closure (ya probado), así el tick nunca lanza: un
 * render roto se salta y el siguiente tick lo corrige. Ante render inicial
 * roto se limpia el timer antes del Err (disposability, sin fugas).
 */
export function startPhraseRotation(
  render: unknown,
  clock?: PhraseClock,
): Result<PhraseHandle, PhraseError> {
  if (!isPhraseRender(render))
    return { ok: false, error: { kind: "BadRender" } };
  const active: PhraseClock = clock ?? defaultPhraseClock();
  let index = 0;
  const tick = (): void => {
    const next = nextPhraseIndex(index);
    if (!next.ok) return;
    index = next.value;
    const phrase = phraseAt(index);
    if (!phrase.ok) return;
    try {
      render(phrase.value);
    } catch {
      // Tick best-effort: se ignora, el siguiente corrige (igual que los
      // eventos WS malformados que index.astro ya tolera).
    }
  };
  let timerId: unknown;
  try {
    timerId = active.setInterval(tick, PHRASE_INTERVAL_MS);
  } catch {
    return { ok: false, error: { kind: "BadClock" } };
  }
  const first = phraseAt(index);
  if (!first.ok) {
    try {
      active.clearInterval(timerId);
    } catch {
      // Limpieza best-effort camino al Err.
    }
    return { ok: false, error: first.error };
  }
  try {
    render(first.value);
  } catch {
    try {
      active.clearInterval(timerId);
    } catch {
      // Limpieza best-effort camino al Err.
    }
    return { ok: false, error: { kind: "BadRender" } };
  }
  return { ok: true, value: { timerId, clock: active } };
}

/**
 * Detiene la rotación. Idempotente: doble stop limpia dos veces sin daño.
 * Handle malformado va por Err, nunca throw.
 */
export function stopPhraseRotation(
  handle: unknown,
): Result<void, PhraseError> {
  if (!isPhraseHandle(handle))
    return { ok: false, error: { kind: "BadHandle" } };
  try {
    handle.clock.clearInterval(handle.timerId);
  } catch {
    return { ok: false, error: { kind: "BadClock" } };
  }
  return { ok: true, value: undefined };
}
