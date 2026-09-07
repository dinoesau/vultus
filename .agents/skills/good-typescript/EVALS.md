# Good-TypeScript - Evaluaciones

Corre cada escenario en sesion fresca con la skill instalada.
Compara contra la baseline: mismo prompt sin la skill.
La skill gana solo si cumple todos los comportamientos esperados.
Re-corre cuando cambie el dominio, no solo cuando cambie la skill.

## Escenario 1: refactor de codigo defensivo

Entrada: un handler que recibe `userId: string` y `amount: number`,
repite los mismos `if` en handler y servicio, y hace `throw new Error`.

Esperado:

- [ ] Crea `Email`, `UserId` y `Cents` como brands con unico parser que retorna `Result`.
- [ ] El core acepta solo tipos probados, sin `if` duplicados aguas abajo.
- [ ] `npx tsc --noEmit`, `npx eslint . --max-warnings 0` y `npx vitest run` pasan.
- [ ] Ningun grep de fuga imprime lineas en `domain` o `core` (`safeParse`, `as Email` fuera de dominio, `throw` en dominio).

## Escenario 2: errores estratificados

Entrada: dominio que hace `throw new Error(string)` y un `catch` unico que mapea todo a 400.

Esperado:

- [ ] Union exhaustiva `DomainError` con discriminante `kind`; sin `throw` en el dominio.
- [ ] `AppError` envuelve infra una vez con `cause`; logs solo en el edge.
- [ ] 400, 404 o 422 por variante de dominio; 500 generico para infra.
- [ ] Agregar una variante rompe el `switch` en compilacion via `assertNever`.

## Escenario 3: type-state en workflow ordenado

Entrada: flags `isSubmitted` e `isPaid` revisados con `if` antes de cada accion.

Esperado:

- [ ] Estados `Order<Draft>` a `Order<Paid>` con llave `StageTag` no exportada.
- [ ] Pagar un draft o pedir recibo antes de tiempo da error de tipo (`@ts-expect-error` lo documenta).
- [ ] Sin `safeParse` repetido: el brand viaja probado por referencia.

## Prueba de disparo

Debe activarse con: "quita las validaciones repetidas en este handler TypeScript",
"modela email como branded type", "haz estos errores exhaustivos".
No debe activarse con: codigo Python o Rust, logica ya tipada,
preguntas de infraestructura sin invariantes de dominio.
Si dispara de mas, estrecha la descripcion. Si no dispara, agrega la frase del usuario.
