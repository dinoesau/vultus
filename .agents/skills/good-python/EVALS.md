# Good-Python - Evaluaciones

Corre cada escenario en sesion fresca con la skill instalada.
Compara contra la baseline: mismo prompt sin la skill.
La skill gana solo si cumple todos los comportamientos esperados.
Re-corre cuando cambie el dominio, no solo cuando cambie la skill.

## Escenario 1: refactor de codigo defensivo

Entrada: un handler que recibe `dict` y `Any`, repite `if` e `isinstance`
en handler, servicio y repo, y hace `raise ValueError`.

Esperado:

- [ ] Crea `Email`, `UserId` y `Cents` como dataclasses `frozen=True, slots=True` con unico `parse` que retorna `Result`.
- [ ] El core acepta solo tipos probados, sin `if` duplicados aguas abajo.
- [ ] `mypy --strict .`, `ruff check .` y `pytest` pasan.
- [ ] Ningun grep de fuga imprime lineas en `domain` o `core` (`model_validate`, `isinstance`, `raise ValueError`, `except Exception`).

## Escenario 2: errores estratificados

Entrada: dominio que hace `raise ValueError(str)` y un `except Exception` unico que mapea todo a 400.

Esperado:

- [ ] Union exhaustiva `DomainError` de dataclasses frozen; sin `raise` por outcomes de negocio.
- [ ] `DbError` o `GatewayError` envuelven infra una vez con `cause`; logs solo en el edge.
- [ ] 400, 404 o 422 por variante de dominio; 500 generico para infra.
- [ ] El dominio no importa FastAPI ni logging.

## Escenario 3: type-state en workflow ordenado

Entrada: flags `is_submitted` e `is_paid` revisados con `if` antes de cada accion.

Esperado:

- [ ] Estados `OrderState[Draft]` a `OrderState[Paid]` con genericos de etapa.
- [ ] Pagar un draft da error en `mypy --strict` y `pyright`.
- [ ] Rehidratacion desde DB usa red runtime minima (`rehydrate_paid` con `Result`).
- [ ] Sin `model_validate` repetido: el value object viaja probado por referencia.

## Prueba de disparo

Debe activarse con: "quita las validaciones repetidas en este handler Python",
"modela email como value object", "haz estos errores exhaustivos".
No debe activarse con: codigo TypeScript o Rust, logica ya tipada,
preguntas de infraestructura sin invariantes de dominio.
Si dispara de mas, estrecha la descripcion. Si no dispara, agrega la frase del usuario.
