---
name: good-python
description: >
  Esta skill debe usarse cuando el usuario pide "quitar validaciones repetidas en Python",
  "tipar el dominio" o "errores exhaustivos", o menciona "parse don't validate",
  "value objects", "smart constructor", "Result", "type-state", "Hypothesis" o "Pydantic"
  en codigo Python. Convierte validacion runtime repetida en garantias del checker con
  value objects frozen, ADTs, funciones totales y errores estratificados. Solo aplica a Python.
license: MIT
allowed-tools: Bash
metadata:
  version: "1.0.0"
---

# Good-Python - Modelado de dominio funcional en Python

Convierte validacion repetida en runtime en pruebas hechas por el checker.
Parsea una vez en el borde y deja que el core componga tipos que ya son correctos.

## Cuando usar esta skill

Usa esta skill cuando el codigo Python muestre estos sintomas.
Repite los mismos `if` e `isinstance` sobre `dict` o `str` en varias capas.
Las firmas reciben `Any` y nadie sabe que ya fue probado.
Los errores son `str` o `Exception` generica y no se pueden clasificar por programa.
El dominio hace `raise ValueError` y pierde exhaustividad.
Hay `model_validate` repetido en handler, servicio y repo para el mismo valor.
Hay booleanos como `is_paid` que se revisan en orden manual.
El hot path reparsea un valor ya probado.

No uses esta skill para TypeScript o Rust.
No la uses para logica ya modelada con tipos probados.
No la uses para decisiones de infraestructura sin invariantes de dominio.

## Workflow

Copia este checklist en tu respuesta y marca avance.

```
Progreso Good-Python:
- [ ] 1. Parsear en el borde a tipos probados
- [ ] 2. Modelar con value objects y smart constructor
- [ ] 3. Totalizar con Result y railway
- [ ] 4. Estratificar errores por audiencia
- [ ] 5. Verificar con checker y tests
```

### 1. Parsear en el borde, nunca validar en el core

Convierte `dict`, `str` y `Any` en `Email`, `UserId` y `Cents` en cuanto cruzan Pydantic, FastAPI o queue consumers.
El core solo acepta tipos probados.
Si una regla vive en un tipo, borra todo `if` que la rechequee aguas abajo.
Ver [REFERENCE.md](./REFERENCE.md#2-parse-dont-validate).

### 2. Modelar con value objects y smart constructor

Crea un modulo por concepto con dataclass `frozen=True, slots=True` y campo `_value` por convencion.
Expon un unico smart constructor `parse` que retorna `Result`.
`NewType` solo documenta, nunca impone: el enforcement es el value object mas review mas checker.
Ver [REFERENCE.md](./REFERENCE.md#3-pilar-1-value-objects-y-smart-constructors).

Plantilla estricta (baja libertad, seguir literal):

```python
@dataclass(frozen=True, slots=True)
class Email:
    """Solo via parse_email."""
    _value: str

def parse_email(raw: object) -> Result[Email, EmailError]:
    # Validar aqui una sola vez.
    # Retornar Ok(Email(_value=trimmed)) solo aqui.
```

### 3. Totalizar con ADTs y railway

Modela alternativas con uniones de dataclasses frozen y combinaciones con dataclasses producto.
Haz cada funcion parcial una funcion total que retorna `Result`.
Compon con `and_then`, `map_result`, `map_err` o retorno temprano en vez de piramides de `if`.
Nunca uses `assert` para invariantes: desaparece con `python -O`.
Ver [REFERENCE.md](./REFERENCE.md#4-pilar-2-adts-y-funciones-totales).

### 4. Estratificar errores por audiencia

Si defines el dominio, usa union exhaustiva `DomainError` de dataclasses frozen.
Si corres la app, envuelve infra una vez en `DbError` o `GatewayError` con `cause`.
Mapea a HTTP solo en el edge con `domain_to_status` y `report_app_error`.
Nunca retornes `str` ni hagas `raise` por outcomes de negocio desde el dominio.
El dominio nunca importa FastAPI ni logging.
Ver [REFERENCE.md](./REFERENCE.md#6-pilar-4-errores-estratificados).

### 5. Verificar antes de entregar

Ejecuta este loop y repite hasta que pase todo.
Si algo falla, corrige y vuelve a correr desde el paso 1.
Valida cambios contra [EVALS.md](./EVALS.md): corre los 3 escenarios y exige mejora contra la baseline sin skill.

```bash
mypy --strict .
ruff check .
pytest
```

Busca fugas del patron con estos greps.
Si alguno imprime lineas en `domain` o `core`, corrige antes de entregar.

```bash
grep -rn 'model_validate' src/domain src/core || true
grep -rn 'isinstance' src/domain src/core || true
grep -rn 'raise ValueError' src/domain src/core || true
grep -rn 'except Exception' src/domain src/core || true
```

## Decisiones condicionales

Si creas contenido nuevo, sigue el workflow completo en orden.
Si editas codigo existente, empieza por el paso 1 solo en el borde tocado.
Si el error es de negocio esperado, agregalo como variante de `DomainError`.
Si el error es operativo con causa externa, envuelvelo en `AppError` con `cause`.
Si el workflow tiene dos o mas estados ordenados con distintas operaciones, usa type-state.
Si es un solo booleano sin orden, no uses type-state.
Si el hot path ya tiene el valor probado, pasa el value object sin revalidar.
Si debes mecanizar reglas repetidas, usa decorador que deje la regla visible en el modulo de dominio.

## Tabla anti-racionalizacion

| Excusa | Realidad |
|---|---|
| "Es solo un `dict` rapido" | Ese atajo crea la proxima edicion shotgun en tres capas |
| "`NewType` ya me protege" | `NewType` se forja con una llamada; el value object mas review es la barrera |
| "Un `bool is_paid` basta" | El orden se puede olvidar; el type-state lo rechaza el checker |
| "`assert` aqui nunca falla" | `assert` desaparece con `-O`; retorna `Result` para inputs de usuario |
| "Repito `model_validate` por seguridad" | En hot path parsea una vez y pasa el objeto con `slots` sin revalidar |

## Referencias de un nivel

Lee solo lo necesario para la tarea actual.
No sigas links anidados mas alla de estos archivos.

- Patron completo y codigo: [REFERENCE.md](./REFERENCE.md).
- Pares before/after copiables: [EXAMPLES.md](./EXAMPLES.md).
- Casos de evaluacion: [EVALS.md](./EVALS.md).
