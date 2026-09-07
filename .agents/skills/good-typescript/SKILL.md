---
name: good-typescript
description: >
  Esta skill debe usarse cuando el usuario pide "quitar validaciones repetidas en
  TypeScript", "tipar el dominio" o "errores exhaustivos", o menciona "parse don't
  validate", "branded types", "smart constructor", "Result", "type-state", "fast-check"
  o "Zod" en codigo TypeScript. Convierte validacion runtime repetida en garantias del
  compilador con brands zero-runtime, ADTs, funciones totales y errores estratificados.
  Solo aplica a TypeScript.
license: MIT
allowed-tools: Bash
metadata:
  version: "1.0.0"
---

# Good-TypeScript - Modelado de dominio funcional en TypeScript

Convierte validacion repetida en runtime en pruebas hechas por el compilador.
Parsea una vez en el borde y deja que el core componga tipos que ya son correctos.

## Cuando usar esta skill

Usa esta skill cuando el codigo TypeScript muestre estos sintomas.
Repite los mismos `if` sobre `string` o `number` en varias capas.
Las firmas reciben primitivos y nadie sabe que ya fue probado.
Los errores son `string` o `unknown` y no se pueden clasificar por programa.
El dominio hace `throw new Error` y pierde exhaustividad.
Hay `safeParse` repetido en handler, servicio y repo para el mismo valor.
Hay booleanos como `isPaid` que se revisan en orden manual.
El hot path repite `safeParse` sobre un valor ya probado.

No uses esta skill para Python o Rust.
No la uses para logica ya modelada con tipos probados.
No la uses para decisiones de infraestructura sin invariantes de dominio.

## Workflow

Copia este checklist en tu respuesta y marca avance.

```
Progreso Good-TypeScript:
- [ ] 1. Parsear en el borde a tipos probados
- [ ] 2. Modelar con branded types y smart constructor
- [ ] 3. Totalizar con Result y railway
- [ ] 4. Estratificar errores por audiencia
- [ ] 5. Verificar con compilador y tests
```

### 1. Parsear en el borde, nunca validar en el core

Convierte `string`, `number` y `unknown` en `Email`, `UserId` y `Cents` en cuanto cruzan Zod, Hono o Fastify.
El core solo acepta tipos probados.
Si una regla vive en un tipo, borra todo `if` que la rechequee aguas abajo.
Ver [REFERENCE.md](./REFERENCE.md#2-parse-dont-validate).

### 2. Modelar con branded types y smart constructor

Crea un modulo por concepto con `Brand<string, "Email">` y un unico parser `parseEmail` que retorna `Result`.
Nunca exportes la llave del brand ni el `as` fuera del modulo.
Agrega accesores de lectura (`emailToString`, `centsToNumber`) sin via de forja.
Ver [REFERENCE.md](./REFERENCE.md#3-pilar-1-branded-types-y-smart-constructors).

Plantilla estricta (baja libertad, seguir literal):

```ts
declare const EmailBrand: unique symbol;
export type Email = string & { readonly [EmailBrand]: "Email" };

export function parseEmail(raw: unknown): Result<Email, EmailError> {
  // Validar aqui una sola vez.
  // Retornar { ok: true, value: trimmed as Email } solo aqui.
}

export function emailToString(email: Email): string {
  return email;
}
```

### 3. Totalizar con ADTs y railway

Modela alternativas con uniones discriminadas y combinaciones con interfaces.
Haz cada funcion parcial una funcion total que retorna `Result`.
Compon con `andThen`, `map`, `mapErr` o retorno temprano en vez de piramides de `if`.
Ver [REFERENCE.md](./REFERENCE.md#4-pilar-2-adts-y-funciones-totales).

### 4. Estratificar errores por audiencia

Si defines el dominio, usa union exhaustiva `DomainError` con `kind` discriminante.
Si corres la app, envuelve infra una vez en `AppError` con `cause`.
Mapea a HTTP solo en el edge con `domainToStatus` y `reportAppError`.
Nunca retornes `unknown` ni hagas `throw` de `string` desde el dominio.
Ver [REFERENCE.md](./REFERENCE.md#6-pilar-4-errores-estratificados).

### 5. Verificar antes de entregar

Ejecuta este loop y repite hasta que pase todo.
Si algo falla, corrige y vuelve a correr desde el paso 1.
Valida cambios contra [EVALS.md](./EVALS.md): corre los 3 escenarios y exige mejora contra la baseline sin skill.

```bash
npx tsc --noEmit
npx eslint . --max-warnings 0
npx vitest run
```

Busca fugas del patron con estos greps.
Si alguno imprime lineas en `domain` o `core`, corrige antes de entregar.

```bash
grep -rn 'safeParse' src/domain src/core || true
grep -rn 'as Email\|as Cents\|as UserId' src --include='*.ts' | grep -v 'src/domain/' || true
grep -rn 'throw new Error' src/domain src/core || true
grep -rn 'isValid' src/domain src/core || true
```

## Decisiones condicionales

Si creas contenido nuevo, sigue el workflow completo en orden.
Si editas codigo existente, empieza por el paso 1 solo en el borde tocado.
Si el error es de negocio esperado, agregalo como variante de `DomainError`.
Si el error es operativo con causa externa, envuelvelo en `AppError` con `cause`.
Si el workflow tiene dos o mas estados ordenados con distintas operaciones, usa type-state.
Si es un solo booleano sin orden, no uses type-state.
Si el hot path solo presta el valor, pasa el brand por referencia sin reparsear.
Si debes derivar schemas, extiende con `extend` o `pick` en vez de copiar campos.

## Tabla anti-racionalizacion

| Excusa | Realidad |
|---|---|
| "Es solo un `string` rapido" | Ese atajo crea la proxima edicion shotgun en tres capas |
| "`throw` en todos lados es mas simple" | Borra los casos exhaustivos que el llamador necesita |
| "Un `bool isPaid` basta" | El orden se puede olvidar; el type-state no compila mal |
| "`as Email` aqui nunca falla" | Si es input de usuario, retorna `Result`; reserva `as` al smart constructor |
| "Repito `safeParse` por seguridad" | En hot path parsea una vez y pasa el brand sin revalidar |

## Referencias de un nivel

Lee solo lo necesario para la tarea actual.
No sigas links anidados mas alla de estos archivos.

- Patron completo y codigo: [REFERENCE.md](./REFERENCE.md).
- Pares before/after copiables: [EXAMPLES.md](./EXAMPLES.md).
- Casos de evaluacion: [EVALS.md](./EVALS.md).
