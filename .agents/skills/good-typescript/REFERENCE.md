# Good-TypeScript - Referencia completa

Fuente: guia arquitectonica sobre error handling, invariantes y modelado funcional de dominio en TypeScript.
Cubre el post completo sin recortes: antipatron, cambio de paradigma y los 5 pilares mas ingenieria avanzada y arquitectura.
Lee solo la seccion que necesites para la tarea actual.

## Contenido

- [1. Antipatron del TypeScript defensivo](#1-antipatron-del-typescript-defensivo)
- [2. Parse don't validate](#2-parse-dont-validate)
- [3. Pilar 1: branded types y smart constructors](#3-pilar-1-branded-types-y-smart-constructors)
- [4. Pilar 2: ADTs y funciones totales](#4-pilar-2-adts-y-funciones-totales)
- [5. Pilar 3: Lisp, expresiones y metaprogramacion](#5-pilar-3-lisp-expresiones-y-metaprogramacion)
- [6. Pilar 4: errores estratificados](#6-pilar-4-errores-estratificados)
- [7. Pilar 5: type-state en compilacion](#7-pilar-5-type-state-en-compilacion)
- [8. Ingenieria avanzada: brands zero-runtime y fast-check](#8-ingenieria-avanzada-brands-zero-runtime-y-fast-check)
- [9. Arquitectura: functional core, imperative shell](#9-arquitectura-functional-core-imperative-shell)
- [10. Tabla defensive vs type-driven](#10-tabla-defensive-vs-type-driven)
- [11. Reglas de oro y bibliografia](#11-reglas-de-oro-y-bibliografia)

## 1. Antipatron del TypeScript defensivo

El habito tentador es aceptar `string`, `number` y `unknown` en cada funcion y rechequear en cada capa.
Ninguna firma registra lo ya probado, asi que el mismo payload se valida en handler, servicio y repo.
Type erasure hace este habito sentir responsable. No lo es. Es caro, ruidoso y fragil.

```ts
// Antipatron: primitivos fluyen por todas las capas.
export async function processRefund(req: Request, res: Response): Promise<void> {
  const body: unknown = await req.body;
  if (typeof body !== "object" || body === null) throw new Error("Invalid payload");
  const payload = body as { userId?: unknown; amount?: unknown };
  if (typeof payload.userId !== "string" || payload.userId.trim() === "") {
    throw new Error("Missing UserId");
  }
  if (typeof payload.amount !== "number" || !(payload.amount > 0)) {
    throw new Error("Invalid amount");
  }
  // ... misma copia en sendReceipt, porque string no prueba nada.
}
```

Cada cambio a la regla de email exige edicion shotgun en varias capas.
Repetir `RefundSchema.safeParse()` en cada capa no arregla nada: cambia la forma pero no la arquitectura.
Sigues pagando el parseo en el hot path y acoplas reglas de negocio a infraestructura.
El costo profundo es paranoia: como nada prueba lo ya validado, todo se rechequea.
El `catch` unico que mapea todo a 400 confunde un typo de usuario con una base caida.

El contrato que TypeScript permite es distinto.
Parsea datos no confiables una vez en el borde.
Entrega al core solo tipos que no pueden estar mal por disciplina.
Borra los guardias duplicados para siempre.

## 2. Parse don't validate

Alexis King capturo la idea en `Parse, don't validate` (2019).
Validar inspecciona un valor y conserva el tipo debil.
Parsear consume el tipo debil y produce un tipo fuerte con la prueba incluida.

Validar tiene esta forma: responde una pregunta y tira la respuesta.

```ts
export function isValidEmail(raw: string): boolean {
  const parts = raw.split("@");
  return parts.length === 2 && parts[0] !== "" && parts[1]?.includes(".") === true;
}

export function notify(rawEmail: string): void {
  if (isValidEmail(rawEmail)) {
    // rawEmail sigue siendo string. El compilador no aprendio nada.
    console.log(`sending to ${rawEmail}`);
  }
}
```

Parsear transforma y certifica en un solo movimiento.

```ts
declare const EmailBrand: unique symbol;
export type Email = string & { readonly [EmailBrand]: "Email" };
export type EmailError =
  | { readonly kind: "MissingAt" }
  | { readonly kind: "EmptyLocalPart" }
  | { readonly kind: "InvalidDomain" };
export type Result<T, E> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

export function parseEmail(raw: unknown): Result<Email, EmailError> {
  if (typeof raw !== "string") return { ok: false, error: { kind: "MissingAt" } };
  const trimmed = raw.trim();
  const at = trimmed.indexOf("@");
  if (at < 0) return { ok: false, error: { kind: "MissingAt" } };
  if (trimmed.slice(0, at) === "") return { ok: false, error: { kind: "EmptyLocalPart" } };
  if (!trimmed.slice(at + 1).includes(".")) return { ok: false, error: { kind: "InvalidDomain" } };
  // Unico cast sancionado del codebase. Vive aqui, se revisa una vez, se testea con fast-check.
  return { ok: true, value: trimmed as Email };
}

export function notifyParsed(email: Email): void {
  // Sin chequeo. El tipo es la prueba.
  console.log(`sending to ${email}`);
}
```

La regla de borde es simple.
Los datos cruzan el borde como `string` y `unknown`.
Viajan dentro del core como `Email`, `UserId` y `Cents`.
El parser vive en exactamente un modulo por tipo.
Todo lo que esta detras compone sin guardias.

## 3. Pilar 1: branded types y smart constructors

Eric Evans los llama Value Objects en `Domain-Driven Design`.
TypeScript los modela con branded types mas smart constructor.
La privacidad del modulo mas lint hace la garantia disciplinaria, no fisica.

Base zero-runtime: solo existe en el checker, emite cero JavaScript.

```ts
// domain/brand.ts
export type Brand<T, Name extends string> = T & { readonly __brand: Name };
export type Email = Brand<string, "Email">;
export type UserId = Brand<string, "UserId">;
export type Cents = Brand<number, "Cents">;
```

Cada brand es un literal distinto, asi `Email` no es asignable a `UserId` aunque ambos envuelvan `string`.
Un `string` pelado no es asignable a ninguno. Ese es todo el truco.

Cada smart constructor vive en su modulo y solo exporta el tipo y el parser.

```ts
// domain/email.ts: unico modulo que puede crear Email.
export function parseEmail(raw: unknown): Result<Email, EmailError> { /* ... */ }
export function emailToString(email: Email): string { return email; }
```

```ts
// domain/money.ts: mismo patron para numericos.
export type MoneyError = { readonly kind: "NonPositive"; readonly received: number };
export function parseCents(raw: unknown): Result<Cents, MoneyError> {
  if (typeof raw !== "number" || !Number.isFinite(raw)) {
    return { ok: false, error: { kind: "NonPositive", received: NaN } };
  }
  if (!Number.isInteger(raw) || raw <= 0) {
    return { ok: false, error: { kind: "NonPositive", received: raw } };
  }
  return { ok: true, value: raw as Cents };
}
export function centsToNumber(amount: Cents): number { return amount; }
```

Para agregados con forma de clase, usa llave de tag no exportada para que nadie forje el literal fuera del modulo.

```ts
// domain/order.ts
const OrderTag: unique symbol = Symbol("OrderTag");
export interface Order {
  readonly id: string;
  readonly userId: UserId;
  readonly email: Email;
  readonly amount: Cents;
  readonly method: PaymentMethod;
  readonly [OrderTag]: "Order";
}
export function createOrder(input: {
  readonly id: string; readonly userId: UserId; readonly email: Email;
  readonly amount: Cents; readonly method: PaymentMethod;
}): Order {
  // Inputs ya probados, sin validacion aqui.
  return { ...input, [OrderTag]: "Order" };
}
```

Zod y Effect Schema encajan como implementacion del parser dentro del smart constructor. Son el portero, no el dominio.

```ts
// domain/refund-request.ts: schema como parser, brand como prueba.
const RefundSchema = z.object({
  userId: z.string().uuid(),
  email: z.string(),
  amount: z.number(),
});
export function parseRefundRequest(data: unknown): Result<RefundRequest, RefundRequestError> {
  const shaped = RefundSchema.safeParse(data);
  if (!shaped.success) return { ok: false, error: { kind: "BadShape", issues: shaped.error.message } };
  const email = parseEmail(shaped.data.email);
  if (!email.ok) return { ok: false, error: { kind: "BadEmail", error: email.error } };
  const amount = parseCents(shaped.data.amount);
  if (!amount.ok) return { ok: false, error: { kind: "BadAmount" } };
  return { ok: true, value: { userId: shaped.data.userId as UserId, email: email.value, amount: amount.value } };
}
```

Limite honesto: en Rust el campo privado es inforjable fisicamente. En TypeScript el brand se borra en runtime y cualquier modulo puede escribir `raw as Email`. La inforjabilidad es disciplinaria. Sostenla con tres reglas: el `as` vive solo en el smart constructor, prohibe `as` fuera con `@typescript-eslint/consistent-type-assertions`, y re-exporta el tipo opaco sin la llave del brand. Revisa cada `as` nuevo como un `sudo`.

## 4. Pilar 2: ADTs y funciones totales

Paul Chiusano y Runar Bjarnason ensenan esto en `Functional Programming in Scala`.
Modela con tipos precisos, escribe funciones totales y compon con combinadores.
Las uniones e interfaces de TypeScript son ADTs. `Result` discriminado es tu monada `Either`.

Sum types enumeran alternativas exclusivas.

```ts
// domain/payment.ts
export type PaymentMethod =
  | { readonly kind: "card"; readonly lastFour: string }
  | { readonly kind: "transfer"; readonly iban: string }
  | { readonly kind: "cash" };
```

Product types combinan hechos independientes.

```ts
export interface OrderShape {
  readonly userId: UserId;
  readonly email: Email;
  readonly amount: Cents;
  readonly method: PaymentMethod;
}
```

Sin `null`, sin `method: string` sin estructura, sin objeto a medio construir.
Exhaustividad con helper que toma `never`.

```ts
export function assertNever(value: never, message = "Unhandled case"): never {
  throw new Error(`${message}: ${JSON.stringify(value)}`);
}
export function feeFor(method: PaymentMethod): number {
  switch (method.kind) {
    case "card": return 30;
    case "transfer": return 10;
    case "cash": return 0;
    default: return assertNever(method);
  }
}
```

Agrega `{ kind: "crypto" }` y `feeFor` deja de compilar hasta manejarlo. Esa ruptura es la funcionalidad.

Una funcion total esta definida para el 100 por ciento de sus inputs. Nunca hace throw por casos esperados.

```ts
// Parcial: truena con cero y NaN.
export function refundSharePartial(amount: number, parts: number): number {
  return amount / parts;
}

// Total: cada input mapea a un resultado explicito.
export type SplitError = { readonly kind: "EmptyParts" } | { readonly kind: "NotDivisible" };
export function refundShareTotal(amount: Cents, parts: number): Result<Cents, SplitError> {
  if (!Number.isInteger(parts) || parts <= 0) return { ok: false, error: { kind: "EmptyParts" } };
  const share = amount / parts;
  if (!Number.isInteger(share)) return { ok: false, error: { kind: "NotDivisible" } };
  return { ok: true, value: share as Cents };
}
```

Disciplina `strict` mas `noUncheckedIndexedAccess`: indexar, dividir y leer JSON son parciales, asi que cada uno retorna `Result` o estrecha antes de usarse.

Composicion con `map`, `andThen` y `mapErr` en vez de piramides. Esto es Railway Oriented Programming de Scott Wlaschin.

```ts
// domain/result.ts
export function map<T, U, E>(result: Result<T, E>, fn: (value: T) => U): Result<U, E> {
  return result.ok ? { ok: true, value: fn(result.value) } : result;
}
export function andThen<T, U, E, F>(
  result: Result<T, E>, fn: (value: T) => Result<U, F>,
): Result<U, E | F> {
  return result.ok ? fn(result.value) : result;
}
export function mapErr<T, E, F>(result: Result<T, E>, fn: (error: E) => F): Result<T, F> {
  return result.ok ? result : { ok: false, error: fn(result.error) };
}
```

Para cadenas largas, el retorno temprano manual es el `?` de TypeScript con la misma semantica monadica.

```ts
export function buildOrderClean(rawEmail: unknown, rawAmount: unknown): Result<OrderShape, string> {
  const email = parseEmail(rawEmail);
  if (!email.ok) return { ok: false, error: `bad email: ${email.error.kind}` };
  const amount = parseCents(rawAmount);
  if (!amount.ok) return { ok: false, error: "bad amount" };
  return {
    ok: true,
    value: {
      userId: "00000000-0000-4000-8000-000000000000" as OrderShape["userId"],
      email: email.value, amount: amount.value, method: { kind: "cash" },
    },
  };
}
```

## 5. Pilar 3: Lisp, expresiones y metaprogramacion

TypeScript hereda su alma de expresion de su linaje ML: ternarios, `switch` como expresion via helpers y `match` de `ts-pattern` retornan valores asignables directo.
Los schemas son la segunda mitad: un schema Zod es un AST manipulable que puedes inspeccionar, componer y del cual generas codigo.

Prefiere expresiones sobre statements al construir valores de dominio.

```ts
import { match } from "ts-pattern";

export function labelFor(amount: number): string {
  const tier = amount > 100_000 ? "enterprise" : amount > 1_000 ? "standard" : "micro";
  return tier;
}

export function describeResult(result: Result<Email, { kind: string }>): string {
  return match(result)
    .with({ ok: true }, ({ value }) => `valid: ${value}`)
    .with({ ok: false }, ({ error }) => `invalid: ${error.kind}`)
    .exhaustive();
}
```

Sin danza de `let result;`. Sin variable sin inicializar. `.exhaustive()` rompe el build cuando la union crece.

Codigo como dato en dos lugares: schemas que manipulas y tipos que computan.

```ts
const BaseRefundDto = z.object({
  orderId: z.string().uuid(),
  email: z.string(),
  amountCents: z.number(),
});
export const RefundDto = BaseRefundDto.extend({ idempotencyKey: z.string().uuid() });
export const RefundAmountOnly = BaseRefundDto.pick({ amountCents: true });
```

```ts
type EmailString = `${string}@${string}.${string}`;
const policy = { maxCents: 500_000, currency: "USD" } as const satisfies {
  readonly maxCents: number; readonly currency: string;
};
```

`EmailString` es documentacion, no prueba: los template literal types no revisan `trim()` ni rangos, y se borran en runtime. Usalos para autocomplete, pero el smart constructor es el unico punto de enforcement.

Mecaniza repeticion con helpers pequenos, nunca con logica de negocio escondida.

```ts
export function makeStringBrand<Name extends string>(name: Name, schema: z.ZodString) {
  return {
    schema: schema.transform((value) => value as Brand<string, Name>),
    parse(raw: unknown): Result<Brand<string, Name>, { readonly kind: string; readonly name: Name }> {
      const parsed = schema.safeParse(raw);
      if (!parsed.success) return { ok: false, error: { kind: "invalid", name } };
      return { ok: true, value: parsed.data as Brand<string, Name> };
    },
  };
}
export const EmailParser = makeStringBrand("Email", z.string().trim().min(3));
```

Regla de helpers: pueden quitar boilerplate de `safeParse`, `trim` y `transform`, pero el invariante queda visible en el modulo de dominio. Si el revisor no ve la regla del email sin abrir el helper, la abstraccion fue demasiado lejos.

## 6. Pilar 4: errores estratificados

No todos los errores pertenecen al mismo tipo.
Los de dominio son resultados de negocio esperados y deben ser exhaustivos.
Los de infraestructura son fallas operativas y necesitan cadenas `cause`.
Mezclarlos en un `string` o un `unknown` lanzado destruye la senal.
Estratifica en tres capas: core expone dominio, app envuelve infra una vez, edge reporta con contexto.

```ts
// domain/errors.ts: vocabulario exhaustivo de negocio.
export type DomainError =
  | { readonly kind: "InvalidEmail"; readonly error: EmailError }
  | { readonly kind: "InvalidAmount" }
  | { readonly kind: "UserNotFound"; readonly userId: string }
  | { readonly kind: "InsufficientFunds"; readonly requested: number; readonly balance: number }
  | { readonly kind: "AlreadyRefunded"; readonly orderId: string };
```

El `switch` exhaustivo fuerza decisiones de producto.

```ts
export function domainToStatus(error: DomainError): number {
  switch (error.kind) {
    case "InvalidEmail":
    case "InvalidAmount": return 400;
    case "UserNotFound": return 404;
    case "InsufficientFunds":
    case "AlreadyRefunded": return 422;
    default: return assertNever(error);
  }
}
export function domainToMessage(error: DomainError): string {
  switch (error.kind) {
    case "InvalidEmail": return `invalid email: ${error.error.kind}`;
    case "InvalidAmount": return "invalid amount: must be a positive integer";
    case "UserNotFound": return "user not found";
    case "InsufficientFunds": return "insufficient funds";
    case "AlreadyRefunded": return `refund already processed for order ${error.orderId}`;
    default: return assertNever(error);
  }
}
```

Envuelve infraestructura una vez en la capa de aplicacion con `cause` explicito.

```ts
// app/errors.ts
export type AppError =
  | { readonly kind: "Domain"; readonly error: DomainError }
  | { readonly kind: "Database"; readonly cause: unknown }
  | { readonly kind: "Gateway"; readonly cause: unknown };
```

Agrega contexto y logs solo en el edge, donde humanos leen.

```ts
export function reportAppError(
  error: AppError, logger: { error: (message: string, details?: unknown) => void },
): { readonly status: number; readonly body: { readonly error: string } } {
  if (error.kind === "Domain") {
    return { status: domainToStatus(error.error), body: { error: domainToMessage(error.error) } };
  }
  logger.error("infrastructure failure", { kind: error.kind, cause: error.cause });
  return { status: 500, body: { error: "internal error" } };
}
```

Tres reglas: nunca retornes `unknown` desde el dominio, nombra la union. Nunca hagas `throw` de `string` o `Error` pelado desde el dominio, retorna `Result<T, DomainError>`. Nunca dejes que el dominio importe el logger o tipos del framework: la flecha apunta del shell al core, nunca al reves.

## 7. Pilar 5: type-state en compilacion

Algunos invariantes no son sobre valores solos sino sobre secuencias. Una orden no se paga antes de enviarse. Un refund no se emite dos veces. Los booleanos `isSubmitted` se olvidan o se revisan en mal orden. Type-state codifica el workflow en genericos para que secuencias malas no compilen.

```ts
// domain/order-lifecycle.ts
export interface Draft { readonly stage: "draft"; }
export interface Submitted { readonly stage: "submitted"; }
export interface Paid { readonly stage: "paid"; }
export type OrderStage = Draft | Submitted | Paid;
declare const StageTag: unique symbol;
export interface Order<S extends OrderStage> {
  readonly id: string;
  readonly amount: Cents;
  readonly stage: S["stage"];
  readonly [StageTag]: S;
}
export function createDraftOrder(id: string, amount: Cents): Order<Draft> {
  return { id, amount, stage: "draft", [StageTag]: { stage: "draft" } as Draft };
}
export function submitOrder(order: Order<Draft>): Order<Submitted> {
  return { id: order.id, amount: order.amount, stage: "submitted", [StageTag]: { stage: "submitted" } as Submitted };
}
export function payOrder(order: Order<Submitted>): Order<Paid> {
  return { id: order.id, amount: order.amount, stage: "paid", [StageTag]: { stage: "paid" } as Paid };
}
export function receiptFor(order: Order<Paid>): string {
  return `paid ${order.amount} for ${order.id}`;
}
```

Uso correcto fluye por el compilador. Transiciones ilegales no compilan.

```ts
const draft = createDraftOrder("ord_1", 5000 as Cents);
const submitted = submitOrder(draft);
const paid = payOrder(submitted);
// @ts-expect-error: payOrder necesita Order<Submitted>.
const illegal = payOrder(draft);
// @ts-expect-error: receipt necesita Order<Paid>.
const early = receiptFor(submitted);
```

TypeScript no destruye el binding viejo `draft` como Rust con move. Sosten el patron con disciplina: prefiere sombrear (`const order = submitOrder(order)`), mantene la llave `StageTag` sin exportar para que nadie forje `Order<Paid>` a mano, y revisa reuso post-transicion en modulos pequenos. Honestidad: el compilador prueba el valor nuevo, solo el review prueba que el binding viejo se descarto.

Usa type-state cuando la secuencia importa y el costo de transicion mala es alto: pagos, provisioning, publicacion, onboarding multi-paso. Heuristica: dos o mas estados ordenados con distintas operaciones. No lo uses para cada booleano o el ruido generico ahoga el dominio.

## 8. Ingenieria avanzada: brands zero-runtime y fast-check

Los brands son abstraccion genuinamente zero-cost: `type Email = string & { __brand: "Email" }` no emite JavaScript. Sin wrapper, sin alocacion, sin indireccion en el hot path. La prueba vive en el checker y desaparece del bundle.

Regla de performance: parsea una vez en el edge, luego pasa el brand por referencia. Nunca llames `safeParse` de nuevo en servicio y repo para un valor ya brandeado.

```ts
// Mal: re-parsear un valor probado en el hot path.
export function chargeTwice(rawAmount: unknown): void {
  const first = AmountSchema.safeParse(rawAmount);
  if (!first.success) return;
  const second = AmountSchema.safeParse(first.data); // Trabajo redundante.
  if (!second.success) return;
}

// Bien: parsea una vez, pasa el brand.
export function chargeOnce(rawAmount: unknown): void {
  const parsed = parseCents(rawAmount);
  if (!parsed.ok) return;
  applyCharge(parsed.value);
}
function applyCharge(_amount: Cents): void { /* hot path: cero checks */ }
```

La logica de parsing merece tests mas fuertes que ejemplos a mano. `fast-check` lanza cientos de inputs sinteticos al smart constructor, incluyendo Unicode y longitudes patologicas.

```ts
// domain/email.properties.test.ts
import * as fc from "fast-check";
test("valid shaped emails always parse", () => {
  fc.assert(fc.property(
    fc.tuple(
      fc.stringOf(fc.constantFrom(..."abcdefghijklmnopqrstuvwxyz0123456789"), { minLength: 1, maxLength: 16 }),
      fc.stringOf(fc.constantFrom(..."abcdefghijklmnopqrstuvwxyz"), { minLength: 1, maxLength: 8 }),
      fc.stringOf(fc.constantFrom(..."abcdefghijklmnopqrstuvwxyz"), { minLength: 2, maxLength: 4 }),
    ),
    ([local, domain, tld]) => { expect(parseEmail(`${local}@${domain}.${tld}`).ok).toBe(true); },
  ));
});
test("parse never throws on arbitrary unicode", () => {
  fc.assert(fc.property(fc.fullUnicodeString(), (raw) => {
    expect(() => parseEmail(raw)).not.toThrow();
  }));
});
```

Corre con `vitest run` y guarda el seed que falla como test de regresion. Ganas robustez matematica: formas validas siempre pasan, invalidas siempre fallan, Unicode hostil nunca lanza, y la normalizacion hace round-trip.

## 9. Arquitectura: functional core, imperative shell

Gary Bernhardt lo resumo en una linea: Functional Core, Imperative Shell.
El core es puro, sincrono y total. Toma tipos de dominio y retorna `Result`. Sin `fetch`, sin sockets, sin reloj, sin `async`. El shell es delgado y con efectos. Habla HTTP y JSON, parsea en el borde, llama al core y mapea errores tipados a status codes.

Define el core puro primero.

```ts
// core/refunds.ts: puro, sync, sin IO.
export interface RefundPolicy { readonly maxCents: number; }
export interface Refund { readonly orderId: string; readonly amount: Cents; }
export interface OrderSnapshot { readonly orderId: string; readonly balance: number; readonly alreadyRefunded: boolean; }
export function calculateRefund(
  order: OrderSnapshot, requested: Cents, policy: RefundPolicy,
): Result<Refund, DomainError> {
  if (order.alreadyRefunded) {
    return { ok: false, error: { kind: "AlreadyRefunded", orderId: order.orderId } };
  }
  if (requested > order.balance) {
    return { ok: false, error: { kind: "InsufficientFunds", requested, balance: order.balance } };
  }
  if (requested > policy.maxCents) {
    return { ok: false, error: { kind: "InvalidAmount" } };
  }
  return { ok: true, value: { orderId: order.orderId, amount: requested } };
}
```

Los DTOs Zod se quedan tontos y crudos en el shell. Sin reglas de negocio.

```ts
// shell/dto.ts
export const RefundRequestDto = z.object({
  orderId: z.string().uuid(),
  email: z.string(),
  amountCents: z.number(),
});
```

El handler Hono une los dos mundos y nada mas. En Fastify reemplaza `c.req.json()` por `request.body` y `c.json()` por `reply.code().send()`. El core no cambia porque nunca importo el framework.

```ts
// shell/handlers.ts
app.post("/refund", async (c) => {
  // 1. Parsear en el borde: unknown JSON se vuelve brands probados.
  const raw: unknown = await c.req.json().catch(() => null);
  const shaped = RefundRequestDto.safeParse(raw);
  if (!shaped.success) return c.json({ error: shaped.error.message }, 400);
  const email = parseEmail(shaped.data.email);
  if (!email.ok) {
    const e: DomainError = { kind: "InvalidEmail", error: email.error };
    return c.json({ error: domainToMessage(e) }, domainToStatus(e));
  }
  const amount = parseCents(shaped.data.amountCents);
  if (!amount.ok) {
    const e: DomainError = { kind: "InvalidAmount" };
    return c.json({ error: domainToMessage(e) }, domainToStatus(e));
  }
  // 2. Rehidratar estado minimo, llamar al core puro.
  const refund = calculateRefund(
    { orderId: shaped.data.orderId, balance: 10_000, alreadyRefunded: false },
    amount.value, { maxCents: 500_000 },
  );
  if (!refund.ok) return c.json({ error: domainToMessage(refund.error) }, domainToStatus(refund.error));
  // 3. Mapear a transporte. Sin logica aqui.
  return c.json({ orderId: refund.value.orderId, refundedCents: refund.value.amount }, 200);
});
```

El testing se divide limpio. Prueba `calculateRefund` con structs planos y sin mocks. Prueba el handler con payloads JSON reales: JSON malformado, email malo, monto negativo y doble refund, cada uno con su status.

## 10. Tabla defensive vs type-driven

Guarda esta tabla como checklist de review.
Si una fila se mueve a la izquierda, regresa la prueba al tipo.

| Concepto | Defensive TypeScript | Type-Driven TypeScript | Beneficio |
|---|---|---|---|
| Parseo de borde | `if` repetidos en cada funcion sobre `string` crudo | `parseEmail(unknown)` retorna `Result<Email, EmailError>` una vez | Una sola fuente de verdad, cero rechequeos en el core |
| Branded types | Alias de `string` forjables en cualquier lado | `Brand<string, "Email">` creado solo por el smart constructor, `as` baneado fuera por lint | Inforjabilidad disciplinaria pese a erasure |
| Totalidad | Division e indexado que lanzan o dan `undefined` en bordes | `Cents` mas `Result` fuerza manejo de cero, NaN e indice faltante con `noUncheckedIndexedAccess` | Edge cases como obligacion de compilacion |
| Composicion | Piramides de `if` con `throw` en cada nivel | `andThen`, `map`, `mapErr` y retorno temprano sobre el railway | Happy path lineal con riel de error tipado |
| Errores de dominio | `throw new Error(string)`, atrapado como `unknown` | Union exhaustiva `DomainError`, `switch` mas `assertNever` cubre cada variante | Nuevos casos rompen el build de forma ruidosa |
| Errores de edge | Un solo `catch` que mapea todo a 400 | `AppError` con `cause`, shell mapea dominio a 4xx e infra a 500 con logs | Contexto rico donde humanos leen logs, tipos precisos donde el codigo ramifica |
| Estado de workflow | Flags como `isPaid` con `if` antes de cada accion | Type-state `Order<Draft>` a `Order<Paid>` con genericos y tag | Transiciones ilegales no compilan |
| Costo en hot path | `safeParse` repetido en handler, servicio y repo | Parse una vez en el edge, pasa brands zero-runtime sin alocacion | Prueba sin impuesto de performance |
| Testing | Tests a mano con pocos literales | `fast-check` con cientos de inputs Unicode mas shrinking y seeds | Confianza matematica en parsers, reproductores minimos |
| Arquitectura | Handlers mezclan Zod, DB y reglas con `async` en todos lados | Core puro sync con `calculateRefund` mas shell Hono y Zod delgado | Core testeable y portable, efectos aislados y auditables |

## 11. Reglas de oro y bibliografia

Primera regla: parsea una vez en el borde, nunca valides en el core.
`string` y `unknown` crudos entran por HTTP o queues y se vuelven `Email`, `UserId` y `Cents` de inmediato.
El core solo acepta tipos probados y contiene cero `isValid` y cero `safeParse` repetidos.

Segunda regla: haz estados ilegales irrepresentables y borra los guardias.
Prefiere uniones para alternativas, interfaces para combinaciones y branded types con smart constructors para invariantes.
Si una regla vive en un tipo, quita cada `if` que la rechequee aguas abajo.

Tercera regla: escribe funciones totales y compon sobre el railway.
Retorna `Result` para cada operacion parcial, maneja cada variante y encadena con retorno temprano, `map` y `andThen`.
Reserva `throw` para bugs verdaderamente imposibles e infraestructura, nunca para input de usuario.

Cuarta regla: estratifica errores por audiencia.
El dominio expone uniones exhaustivas `DomainError`.
La app envuelve fallas de infra una vez con `cause`.
El edge agrega contexto humano, logs y mapeo HTTP.
Nunca filtres `unknown` lanzado desde APIs de dominio.

Quinta regla: mete workflows y costos al sistema de tipos.
Usa type-state con genericos para ciclos ordenados.
Usa brands zero-runtime en hot paths.
Cubre parsers con `fast-check` y manten el shell Hono o Fastify delgado alrededor de un core funcional puro.

Bibliografia del post.
Alexis King, `Parse, don't validate` (2019).
Paul Chiusano y Runar Bjarnason, `Functional Programming in Scala` (2014).
Harold Abelson y Gerald Jay Sussman, `Structure and Interpretation of Computer Programs` (1996).
Eric Evans, `Domain-Driven Design` (2003).
Edwin Brady, `Type-Driven Development with Idris` (2017).
Scott Wlaschin, `Railway Oriented Programming` (2013).
Gary Bernhardt, `Functional Core, Imperative Shell` (2012).
