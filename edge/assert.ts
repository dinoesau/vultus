/** Unico sitio con `throw` por bug imposible. El dominio retorna Result, nunca lanza. */
export function assertNever(value: never, message = "Unhandled case"): never {
  throw new Error(`${message}: ${JSON.stringify(value)}`);
}
