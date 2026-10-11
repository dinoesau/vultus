import type { Result } from "./compare-client.ts";

// Seam Uvi (Story-156): visibilidad del tamagotchi de espera.
// Fuente única anti Magic Strings: el id y la clase busy viven aquí;
// index.astro solo importa showUvi/hideUvi/parseUviElement, nunca repite
// los literales. Sin `as` fuera del mint (no hay brand aquí: el modo es
// unión cerrada, no string libre). Todo total, nunca lanza.
export const UVI_ID = "uvi" as const;
export const UVI_BUSY_CLASS = "uvi--job" as const;
export const UVI_TESTID = "uvi" as const;

export type UviMode = "demo" | "job";

export type UviError =
  | { readonly kind: "MissingUvi" }
  | { readonly kind: "WrongNode" };

function isDivElement(found: unknown): found is HTMLDivElement {
  return found instanceof HTMLDivElement;
}

/** Parse-once del nodo #uvi. Ausente o tipo erróneo van por Err, nunca throw. */
export function parseUviElement(root: ParentNode): Result<HTMLDivElement, UviError> {
  const found = root.querySelector(`#${UVI_ID}`);
  if (found === null) return { ok: false, error: { kind: "MissingUvi" } };
  if (!isDivElement(found)) return { ok: false, error: { kind: "WrongNode" } };
  return { ok: true, value: found };
}

/**
 * Muestra a Uvi. `demo` = carga demo sin scanline; `job` = job vivo con
 * scanline (clase busy). Total: muta solo el nodo ya probado.
 */
export function showUvi(el: HTMLDivElement, mode: UviMode): void {
  el.hidden = false;
  el.setAttribute("aria-hidden", "false");
  if (mode === "job") {
    el.classList.add(UVI_BUSY_CLASS);
  } else {
    el.classList.remove(UVI_BUSY_CLASS);
  }
}

/** Oculta a Uvi (idle/done/failed). Total, nunca lanza. */
export function hideUvi(el: HTMLDivElement): void {
  el.hidden = true;
  el.classList.remove(UVI_BUSY_CLASS);
  el.setAttribute("aria-hidden", "true");
}

export function isUviVisible(el: HTMLDivElement): boolean {
  return !el.hidden;
}
