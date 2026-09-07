# Good-Python - Referencia completa

Fuente: guia arquitectonica sobre error handling, invariantes y modelado funcional de dominio en Python.
Cubre el post completo sin recortes: antipatron, cambio de paradigma y los 5 pilares mas ingenieria avanzada y arquitectura.
Lee solo la seccion que necesites para la tarea actual.

## Contenido

- [1. Antipatron del Python defensivo](#1-antipatron-del-python-defensivo)
- [2. Parse don't validate](#2-parse-dont-validate)
- [3. Pilar 1: value objects y smart constructors](#3-pilar-1-value-objects-y-smart-constructors)
- [4. Pilar 2: ADTs y funciones totales](#4-pilar-2-adts-y-funciones-totales)
- [5. Pilar 3: Lisp, expresiones y metaprogramacion](#5-pilar-3-lisp-expresiones-y-metaprogramacion)
- [6. Pilar 4: errores estratificados](#6-pilar-4-errores-estratificados)
- [7. Pilar 5: type-state con el checker](#7-pilar-5-type-state-con-el-checker)
- [8. Ingenieria avanzada: parse-once y Hypothesis](#8-ingenieria-avanzada-parse-once-y-hypothesis)
- [9. Arquitectura: functional core, imperative shell](#9-arquitectura-functional-core-imperative-shell)
- [10. Tabla defensive vs type-driven](#10-tabla-defensive-vs-type-driven)
- [11. Reglas de oro y bibliografia](#11-reglas-de-oro-y-bibliografia)

## 1. Antipatron del Python defensivo

El habito tentador en Python dinamico es aceptar `dict`, `str` y `Any` en cada funcion y rechequear en cada capa.
Como ninguna firma registra lo ya probado, el mismo payload se valida en handler, servicio y repo.

```python
@app.post("/refund")
async def process_refund(req: Request) -> JSONResponse:
    # Antipatron: dict/Any fluye por todas las capas.
    body: object = await req.json()
    if not isinstance(body, dict):
        raise ValueError("Invalid payload")
    if "userId" not in body or not isinstance(body["userId"], str):
        raise ValueError("Missing UserId")
    amount = body.get("amount")
    if not isinstance(amount, (int, float)) or amount <= 0:
        raise ValueError("Invalid amount")
    # ... mismos tres guardias copiados en servicio y repo.
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
```

Una variante mas insidiosa repite `RefundSchema.model_validate()` en cada capa en vez de `if` crudos.
La forma cambio pero la arquitectura no: sigues pagando el parseo en el hot path y acoplas reglas a infraestructura.
El costo profundo es paranoia: como nada prueba lo ya validado, todo se rechequea.
El `except Exception -> 400` confunde un typo de usuario con una base caida: ambos son 400, el outage no se pagea y el cliente reintenta algo que nunca va a funcionar.

El contrato que Python permite es distinto, dentro de limites honestos.
Parsea datos no confiables una vez en el borde.
Entrega al core solo tipos que son incorrectos por violacion de convencion, no por accidente.
Borra los guardias duplicados para siempre.
La garantia es convencion mas checker mas un solo guard runtime en el borde.

## 2. Parse don't validate

Alexis King capturo la idea en `Parse, don't validate` (2019).
Validar inspecciona un valor y conserva el tipo debil.
Parsear consume el tipo debil y produce un tipo fuerte con la prueba incluida.

Validar responde una pregunta y tira la respuesta: despues de `is_valid_email`, el valor sigue siendo `str` y la siguiente funcion debe revisar de nuevo.

```python
def is_valid_email(raw: str) -> bool:
    parts = raw.split("@")
    return len(parts) == 2 and parts[0] != "" and "." in parts[1]

def notify(raw_email: str) -> None:
    if is_valid_email(raw_email):
        # raw_email sigue siendo str. El checker no aprendio nada.
        print(f"sending to {raw_email}")
```

Parsear transforma y certifica en un solo movimiento.

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Ok[T]:
    value: T

@dataclass(frozen=True, slots=True)
class Err[E]:
    error: E

type Result[T, E] = Ok[T] | Err[E]

@dataclass(frozen=True, slots=True)
class MissingAt: pass

@dataclass(frozen=True, slots=True)
class EmptyLocalPart: pass

@dataclass(frozen=True, slots=True)
class InvalidDomain:
    reason: str

type EmailError = MissingAt | EmptyLocalPart | InvalidDomain

@dataclass(frozen=True, slots=True)
class Email:
    """Prueba viviente. Solo via parse_email."""
    _value: str
    def __str__(self) -> str:
        return self._value

def parse_email(raw: object) -> Result[Email, EmailError]:
    if not isinstance(raw, str):
        return Err(MissingAt())
    trimmed = raw.strip()
    at = trimmed.find("@")
    if at < 0:
        return Err(MissingAt())
    if trimmed[:at] == "":
        return Err(EmptyLocalPart())
    if "." not in trimmed[at + 1:]:
        return Err(InvalidDomain(reason="domain must contain a dot"))
    # Unica construccion sancionada del codebase. Vive aqui, se revisa una vez, se testea con Hypothesis.
    return Ok(Email(_value=trimmed))

def notify_parsed(email: Email) -> None:
    # Sin chequeo. El tipo es la prueba.
    print(f"sending to {email}")
```

La regla de borde es simple.
Los datos cruzan el borde como `dict` y `str`.
Viajan dentro del core como `Email`, `UserId` y `Cents`.
El parser vive en exactamente un modulo por tipo.
Todo lo que esta detras compone sin guardias.

## 3. Pilar 1: value objects y smart constructors

Eric Evans los llama Value Objects en `Domain-Driven Design`: pequenos, inmutables, auto-validados, sin identidad mas alla de su valor.
Python los modela con dataclass frozen mas smart constructor. La disciplina del modulo mas el campo privado por convencion da la barrera que el runtime no puede.

Primero entiende por que `NewType` no basta: es la funcion identidad en runtime, se forja con una llamada, no lleva logica de validacion.

```python
from typing import NewType
ValidatedUserId = NewType("ValidatedUserId", str)
PositiveAmount = NewType("PositiveAmount", float)

def charge(user_id: ValidatedUserId, amount: PositiveAmount) -> None: ...
# Se forja instantaneo. El checker confia, el runtime no hace nada.
charge(ValidatedUserId(""), PositiveAmount(-99.0))
```

Usa `NewType` como documentacion de primitivos ya parseados, nunca como enforcement.
El patron real es value object frozen con smart constructor.

```python
@dataclass(frozen=True, slots=True)
class UserId:
    """UUID con marca. Solo via UserId.parse."""
    _value: str

    @classmethod
    def parse(cls, raw: object) -> Result["UserId", str]:
        import uuid
        if not isinstance(raw, str):
            return Err("user id must be a string")
        try:
            uuid.UUID(raw)
        except ValueError:
            return Err(f"invalid uuid: {raw!r}")
        return Ok(cls(_value=raw))

    def __str__(self) -> str:
        return self._value

@dataclass(frozen=True, slots=True)
class Cents:
    """Dinero no negativo en centavos."""
    _value: int

    @classmethod
    def parse(cls, raw: object) -> Result["Cents", str]:
        if isinstance(raw, bool) or not isinstance(raw, int):
            return Err(f"amount must be an int, got {type(raw).__name__}")
        if raw <= 0:
            return Err(f"amount must be positive, got {raw}")
        return Ok(cls(_value=raw))

    def to_int(self) -> int:
        return self._value
```

Propiedades clave: `frozen=True` hace instancias hasheables y no mutables. `slots=True` quita `__dict__`, corta memoria y superficie de inyeccion. `_value` por convencion dice no construir directo. `parse` retorna `Result`, nunca lanza por input esperado.

Pydantic v2 encaja como implementacion del parser dentro del smart constructor, no como el dominio. Usa `Annotated` mas `field_validator` para que la regla quede visible en el sitio de definicion.

```python
from pydantic import BaseModel, ValidationError, field_validator

class _RefundInput(BaseModel):
    user_id: str
    amount_cents: int

    @field_validator("user_id")
    @classmethod
    def _must_be_uuid(cls, v: str) -> str:
        import uuid
        uuid.UUID(v)
        return v

    @field_validator("amount_cents")
    @classmethod
    def _must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("amount_cents must be positive")
        return v

@dataclass(frozen=True, slots=True)
class TrustedRefund:
    user_id: UserId
    amount: Cents

def parse_refund_request(data: object) -> Result[TrustedRefund, list[str]]:
    try:
        raw = _RefundInput.model_validate(data)
    except ValidationError as exc:
        return Err([f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors(include_input=False)])
    user_id = UserId.parse(raw.user_id)
    amount = Cents.parse(raw.amount_cents)
    match (user_id, amount):
        case (Ok(uid), Ok(cents)):
            return Ok(TrustedRefund(user_id=uid, amount=cents))
        case (Err(e), _):
            return Err([e])
        case (_, Err(e)):
            return Err([e])
```

Limite honesto: en Rust el campo privado es inforjable fisicamente. En Python cualquier modulo puede escribir `Email(_value="garbage")`. La inforjabilidad es disciplinaria. Sostenla con tres reglas: construccion directa solo dentro del modulo definidor, prohibida fuera por review y lint, y nunca re-exportes el campo crudo como API publica. Revisa cada construccion directa como un `sudo`.

## 4. Pilar 2: ADTs y funciones totales

Paul Chiusano y Runar Bjarnason ensenan esto en `Functional Programming in Scala`.
Modela con tipos precisos, escribe funciones totales y compon con combinadores.
Las uniones y dataclasses frozen de Python son ADTs. `Result` es tu `Either`.

Sum types enumeran alternativas exclusivas.

```python
@dataclass(frozen=True, slots=True)
class Card:
    kind: Literal["card"] = "card"
    last_four: str = ""

@dataclass(frozen=True, slots=True)
class Transfer:
    kind: Literal["transfer"] = "transfer"
    iban: str = ""

@dataclass(frozen=True, slots=True)
class Cash:
    kind: Literal["cash"] = "cash"

type PaymentMethod = Card | Transfer | Cash
```

Product types combinan hechos independientes.

```python
@dataclass(frozen=True, slots=True)
class OrderShape:
    user_id: UserId
    email: Email
    amount: Cents
    method: PaymentMethod
```

Sin `None` como escape, sin `method: str` sin estructura, sin objeto a medio construir.
Exhaustividad con helper que toma `Never`. Nunca uses `assert` builtin para invariantes: desaparece con `python -O`.

```python
from typing import Never

def assert_never(value: Never) -> Never:
    raise AssertionError(f"unhandled case: {value!r}")

def fee_for(method: PaymentMethod) -> int:
    match method:
        case Card():
            return 30
        case Transfer():
            return 10
        case Cash():
            return 0
```

Agrega `Crypto` y `fee_for` falla el chequeo hasta manejarlo. Con `mypy --strict`, un `match` sin wildcard sobre una union se marca. Esa ruptura es la funcionalidad.

Una funcion total esta definida para el 100 por ciento de sus inputs. Nunca lanza por casos esperados, nunca retorna `None` por sorpresa.

```python
# Parcial: truena con cero.
def refund_share_partial(amount: int, parts: int) -> int:
    return amount // parts

@dataclass(frozen=True, slots=True)
class EmptyParts: pass

@dataclass(frozen=True, slots=True)
class NotDivisible:
    amount: int
    parts: int

type SplitError = EmptyParts | NotDivisible

# Total: cada input mapea a un resultado explicito.
def refund_share_total(amount: Cents, parts: int) -> Result[Cents, SplitError]:
    if parts <= 0:
        return Err(EmptyParts())
    raw = amount.to_int()
    if raw % parts != 0:
        return Err(NotDivisible(amount=raw, parts=parts))
    return Ok(Cents(_value=raw // parts))
```

Composicion con `map_result`, `and_then` y `map_err` en vez de piramides. Esto es Railway Oriented Programming de Scott Wlaschin. Para cadenas largas, Python usa el `?` manual via retorno temprano con la misma semantica.

```python
def map_result(result: Result[T, E], fn: Callable[[T], U]) -> Result[U, E]:
    match result:
        case Ok(value):
            return Ok(fn(value))
        case Err(_):
            return result

def and_then(result: Result[T, E], fn: Callable[[T], Result[U, F]]) -> Result[U, E | F]:
    match result:
        case Ok(value):
            return fn(value)
        case Err(_):
            return result

def build_order_clean(raw_email: object, raw_amount: object) -> Result[OrderShape, str]:
    email = parse_email(raw_email)
    if isinstance(email, Err):
        return Err(f"bad email: {email.error!r}")
    amount = Cents.parse(raw_amount)
    if isinstance(amount, Err):
        return Err(f"bad amount: {amount.error}")
    user = UserId.parse("00000000-0000-4000-8000-000000000000")
    if isinstance(user, Err):
        return Err(user.error)
    return Ok(OrderShape(user_id=user.value, email=email.value, amount=amount.value, method=Cash()))
```

## 5. Pilar 3: Lisp, expresiones y metaprogramacion

Abelson y Sussman celebran en `Structure and Interpretation of Computer Programs` el estilo donde programas se construyen de expresiones que evaluan a valores, y donde el codigo es dato manipulable. Python hereda esa alma con `match`, ternarios y comprehensions que retornan valores asignables directo. La metadata `Annotated` es la segunda mitad: un objeto schema que Pydantic inspecciona en runtime.

Prefiere expresiones sobre statements al construir valores de dominio.

```python
def tier_for(amount_cents: int) -> str:
    return "enterprise" if amount_cents > 100_000 else "standard" if amount_cents > 1_000 else "micro"

def describe_email(result: Result[Email, EmailError]) -> str:
    match result:
        case Ok(value):
            return f"valid: {value}"
        case Err(error):
            return f"invalid: {type(error).__name__}"

def active_emails(raw_items: list[object]) -> list[Email]:
    # La comprehension es el valor. Sin danza de lista mutable.
    return [r.value for r in (parse_email(raw) for raw in raw_items) if isinstance(r, Ok)]
```

Sin danza de `result = None`. Sin variable sin inicializar. El checker verifica que cada rama produzca el tipo declarado.

Codigo como dato en dos lugares: metadata `Annotated` que Pydantic lee, y decoradores que generan parsers.

```python
from pydantic import TypeAdapter
EmailAdapter: TypeAdapter[Email] = TypeAdapter(Email)
```

Un movimiento mas fuerte es un decorador pequeno que mecaniza la forma de prueba en veinte value objects sin esconder la regla.

```python
def branded_str_validator(pattern: str) -> Any:
    import re
    compiled = re.compile(pattern)
    def _validate(v: object) -> str:
        if not isinstance(v, str):
            raise ValueError("must be a string")
        cleaned = v.strip()
        if not compiled.fullmatch(cleaned):
            raise ValueError(f"must match {pattern}")
        return cleaned
    return _validate
```

Aplicado con la regla visible en el call site:

```python
from pydantic import BeforeValidator
SlugRaw = Annotated[str, BeforeValidator(branded_str_validator(r"[a-z0-9-]+"))]

class _SlugInput(BaseModel):
    slug: SlugRaw

@dataclass(frozen=True, slots=True)
class Slug:
    _value: str

    @classmethod
    def parse(cls, raw: object) -> Result["Slug", str]:
        try:
            cleaned = _SlugInput.model_validate({"slug": raw}).slug
        except ValidationError:
            return Err(f"invalid slug: {raw!r}")
        return Ok(cls(_value=cleaned))
```

Regla de decoradores, descriptores y metaclases: el helper puede quitar boilerplate de `strip`, regex y `model_validate`, pero el invariante queda visible en el modulo de dominio. Si el revisor no ve la regla del slug sin abrir el decorador, la abstraccion fue demasiado lejos.

## 6. Pilar 4: errores estratificados

No todos los errores pertenecen al mismo tipo.
Los de dominio son resultados de negocio esperados y deben ser exhaustivos.
Los de infraestructura son fallas operativas y necesitan cadenas de causa.
Mezclarlos en un `str` o un `except Exception` pelado destruye la senal.
Estratifica en tres capas: core expone dominio, app envuelve infra una vez, edge reporta con contexto.

```python
@dataclass(frozen=True, slots=True)
class InvalidEmail:
    detail: EmailError

@dataclass(frozen=True, slots=True)
class InvalidAmount:
    detail: str

@dataclass(frozen=True, slots=True)
class UserNotFound:
    user_id: str

@dataclass(frozen=True, slots=True)
class InsufficientFunds:
    requested: int
    balance: int

@dataclass(frozen=True, slots=True)
class AlreadyRefunded:
    order_id: str

type DomainError = InvalidEmail | InvalidAmount | UserNotFound | InsufficientFunds | AlreadyRefunded
```

El `match` exhaustivo fuerza decisiones de producto.

```python
def domain_to_status(error: DomainError) -> int:
    match error:
        case InvalidEmail() | InvalidAmount():
            return 400
        case UserNotFound():
            return 404
        case InsufficientFunds() | AlreadyRefunded():
            return 422

def domain_to_message(error: DomainError) -> str:
    match error:
        case InvalidEmail(detail=detail):
            return f"invalid email: {type(detail).__name__}"
        case InvalidAmount(detail=detail):
            return f"invalid amount: {detail}"
        case UserNotFound():
            return "user not found"
        case InsufficientFunds(requested=requested, balance=balance):
            return f"insufficient funds: requested {requested}, balance {balance}"
        case AlreadyRefunded(order_id=order_id):
            return f"refund already processed for order {order_id}"
```

Envuelve infraestructura una vez en la capa de aplicacion con causa explicita.

```python
@dataclass(frozen=True, slots=True)
class DbError:
    cause: str

@dataclass(frozen=True, slots=True)
class GatewayError:
    cause: str

type AppError = DomainError | DbError | GatewayError

def app_to_status(error: AppError) -> int:
    match error:
        case DbError() | GatewayError():
            return 500
        case _:
            domain: DomainError = error
            return domain_to_status(domain)
```

Agrega contexto y logs solo en el edge, donde humanos leen.

```python
import logging
logger = logging.getLogger(__name__)

def report_app_error(error: AppError) -> tuple[int, dict[str, str]]:
    match error:
        case DbError(cause=cause) | GatewayError(cause=cause):
            logger.error("infrastructure failure", extra={"kind": type(error).__name__, "cause": cause})
            return 500, {"error": "internal error"}
        case _:
            domain: DomainError = error
            return domain_to_status(domain), {"error": domain_to_message(domain)}
```

Tres reglas: nunca retornes `str` pelado desde el dominio, nombra la union. Nunca hagas `raise` por outcomes de dominio, retorna `Result[T, DomainError]`. Nunca dejes que el dominio importe FastAPI o logging: la flecha apunta del shell al core, nunca al reves. Reserva `raise` para el edge y para asserts internos de bugs que deben ser 500.

## 7. Pilar 5: type-state con el checker

Algunos invariantes no son sobre valores solos sino sobre secuencias. Una orden no se paga antes de enviarse. Un refund no se emite dos veces. Los booleanos `is_submitted` se olvidan o se revisan en mal orden. Type-state codifica el workflow en genericos para que secuencias malas las rechace `mypy` o `pyright`.

```python
from typing import Generic

@dataclass(frozen=True, slots=True)
class Draft:
    stage: Literal["draft"] = "draft"

@dataclass(frozen=True, slots=True)
class Submitted:
    stage: Literal["submitted"] = "submitted"

@dataclass(frozen=True, slots=True)
class Paid:
    stage: Literal["paid"] = "paid"

S = TypeVar("S", bound=object)

@dataclass(frozen=True, slots=True)
class OrderState(Generic[S]):
    order_id: str
    amount: Cents
    state: S

def create_draft(order_id: str, amount: Cents) -> OrderState[Draft]:
    return OrderState(order_id=order_id, amount=amount, state=Draft())

def submit_order(order: OrderState[Draft]) -> OrderState[Submitted]:
    # Consume conceptualmente el draft: el llamador debe soltar el binding viejo.
    return OrderState(order_id=order.order_id, amount=order.amount, state=Submitted())

def pay_order(order: OrderState[Submitted]) -> OrderState[Paid]:
    return OrderState(order_id=order.order_id, amount=order.amount, state=Paid())

# Solo ordenes pagadas exponen recibo.
def receipt_for(order: OrderState[Paid]) -> str:
    return f"paid {order.amount.to_int()} for {order.order_id}"
```

Uso correcto fluye por el checker. Transiciones ilegales son errores estaticos.

```python
def checkout_demo(amount: Cents) -> str:
    draft = create_draft("ord_1", amount)
    submitted = submit_order(draft)
    paid = pay_order(submitted)
    return receipt_for(paid)

def illegal_demo(amount: Cents) -> None:
    draft = create_draft("ord_1", amount)
    paid = pay_order(draft)  # type: ignore[arg-type] -- mypy: necesita OrderState[Submitted]
```

Corre `mypy --strict` y esa linea falla con tipo incompatible. `pyright` reporta lo mismo.
Manten una red runtime minima para datos rehidratados de la DB, donde el checker no ve la fila:

```python
def rehydrate_paid(order_id: str, amount: Cents, stage: object) -> Result[OrderState[Paid], str]:
    if stage != "paid":
        return Err(f"cannot rehydrate paid order from stage {stage!r}")
    return Ok(OrderState(order_id=order_id, amount=amount, state=Paid()))
```

Python no destruye el binding viejo `draft` como Rust con move. Sosten el patron con disciplina: prefiere rebind (`order = submit_order(order)`), manten las transiciones en un modulo pequeno, y nunca forjes `OrderState[Paid]` a mano fuera de el. Honestidad: el checker prueba el valor nuevo, solo el review prueba que el binding viejo se descarto.

Usa type-state cuando la secuencia importa y el costo de transicion mala es alto: pagos, provisioning, publicacion, onboarding multi-paso. Heuristica: dos o mas estados ordenados con distintas operaciones. No lo uses para cada booleano o el ruido generico ahoga el dominio.

## 8. Ingenieria avanzada: parse-once y Hypothesis

`NewType` es abstraccion genuinamente zero-cost: es la funcion identidad en runtime. Las dataclasses frozen con `slots` estan cerca: una alocacion pequena, sin `__dict__`, sin validadores por instancia tras construir. La prueba vive en el tipo y en la unica llamada `parse`, no en checks repetidos.

Regla de performance: parsea una vez en el edge, luego pasa el valor probado por referencia. Nunca llames `model_validate` de nuevo en servicio y repo para un valor que ya es `Cents` o `Email`.

```python
# Mal: re-parsear un valor probado en el hot path.
def charge_twice(raw_amount: object) -> None:
    first = Cents.parse(raw_amount)
    if isinstance(first, Err):
        return
    second = Cents.parse(first.value.to_int())  # Trabajo redundante.
    if isinstance(second, Err):
        return

# Bien: parsea una vez, pasa la marca.
def charge_once(raw_amount: object) -> None:
    parsed = Cents.parse(raw_amount)
    if isinstance(parsed, Err):
        return
    apply_charge(parsed.value)

def apply_charge(_amount: Cents) -> None:
    # Hot path: cero checks, cero alocaciones extra mas alla del objeto.
    pass
```

La logica de parsing merece tests mas fuertes que ejemplos a mano. `Hypothesis` lanza cientos de inputs sinteticos al smart constructor, incluyendo Unicode, caracteres de control y longitudes patologicas.

```python
from hypothesis import example, given
from hypothesis import strategies as st

email_strategy = st.tuples(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=16),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=4),
).map(lambda parts: f"{parts[0]}@{parts[1]}.{parts[2]}")

@given(email_strategy)
def test_valid_shaped_emails_always_parse(raw: str) -> None:
    assert isinstance(parse_email(raw), Ok)

@given(st.text(min_size=1, max_size=32).filter(lambda s: "@" not in s))
def test_missing_at_never_parses(raw: str) -> None:
    assert isinstance(parse_email(raw), Err)

@given(st.text(alphabet=st.characters()))
def test_parse_never_raises_on_arbitrary_unicode(raw: str) -> None:
    # Cualquier input mapea a Ok o Err, nunca lanza.
    assert isinstance(parse_email(raw), (Ok, Err))

@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8))
@example(" alice@example.com ")
def test_parsed_value_is_trimmed_input(local: str) -> None:
    raw = f" {local}@example.com "
    result = parse_email(raw)
    assert isinstance(result, Ok)
    assert str(result.value) == raw.strip()
```

Corre con `pytest` y guarda el seed que falla. Hypothesis reduce al reproductor minimo e imprime el seed. Fijalo con `@example` como regresion. Ganas robustez matematica en vez de cobertura anecdotica.

## 9. Arquitectura: functional core, imperative shell

Gary Bernhardt lo resumo en una linea: Functional Core, Imperative Shell.
El core es puro, sincrono y total. Toma tipos de dominio y retorna `Result`. Sin `await`, sin sockets, sin reloj, sin imports de FastAPI. El shell es delgado y con efectos. Habla HTTP y JSON, parsea en el borde, llama al core y mapea errores tipados a status codes.

Define el core puro primero.

```python
# core/refunds.py: puro, sync, sin IO.
@dataclass(frozen=True, slots=True)
class RefundPolicy:
    max_cents: int

@dataclass(frozen=True, slots=True)
class Refund:
    order_id: str
    amount: Cents

@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    order_id: str
    balance: int
    already_refunded: bool

def calculate_refund(
    order: OrderSnapshot, requested: Cents, policy: RefundPolicy,
) -> Result[Refund, DomainError]:
    # Pura: todos los inputs ya son tipos probados.
    if order.already_refunded:
        return Err(AlreadyRefunded(order_id=order.order_id))
    if requested.to_int() > order.balance:
        return Err(InsufficientFunds(requested=requested.to_int(), balance=order.balance))
    if requested.to_int() > policy.max_cents:
        return Err(InvalidAmount(detail="exceeds policy maximum"))
    return Ok(Refund(order_id=order.order_id, amount=requested))
```

Los DTOs Pydantic se quedan tontos y crudos en el shell. Sin reglas de negocio.

```python
# shell/dto.py
class RefundRequestDto(BaseModel):
    order_id: str
    email: str
    amount_cents: int
```

El handler FastAPI une los dos mundos y nada mas.

```python
# shell/handlers.py
shell_app = FastAPI()

@shell_app.post("/refund")
async def refund_handler(payload: object) -> JSONResponse:
    # 1. Parsear en el borde: JSON desconocido se vuelve marcas probadas.
    try:
        shaped = RefundRequestDto.model_validate(payload)
    except ValidationError as exc:
        return JSONResponse({"error": exc.errors(include_input=False)}, status_code=400)
    email = parse_email(shaped.email)
    if isinstance(email, Err):
        err: DomainError = InvalidEmail(detail=email.error)
        return JSONResponse({"error": domain_to_message(err)}, status_code=domain_to_status(err))
    amount = Cents.parse(shaped.amount_cents)
    if isinstance(amount, Err):
        err = InvalidAmount(detail=amount.error)
        return JSONResponse({"error": domain_to_message(err)}, status_code=domain_to_status(err))
    # 2. Rehidratar estado minimo, llamar al core puro.
    order = OrderSnapshot(order_id=shaped.order_id, balance=10_000, already_refunded=False)
    refund = calculate_refund(order, amount.value, RefundPolicy(max_cents=500_000))
    if isinstance(refund, Err):
        return JSONResponse(
            {"error": domain_to_message(refund.error)},
            status_code=domain_to_status(refund.error),
        )
    # 3. Mapear a transporte. Sin logica aqui.
    return JSONResponse(
        {"orderId": refund.value.order_id, "refundedCents": refund.value.amount.to_int()},
        status_code=200,
    )
```

El testing se divide limpio. Prueba `calculate_refund` con structs planos y sin mocks. Prueba el handler con payloads JSON reales: JSON malformado, email malo, monto negativo y doble refund, cada uno con su status.

## 10. Tabla defensive vs type-driven

Guarda esta tabla como checklist de review.
Si una fila se mueve a la izquierda, regresa la prueba al tipo.

| Concepto | Defensive Python | Type-Driven Python | Beneficio |
|---|---|---|---|
| Parseo de borde | `if` e `isinstance` repetidos en cada funcion sobre `dict` crudo | `parse_email(object)` retorna `Result[Email, EmailError]` una vez | Una sola fuente de verdad, cero rechequeos en el core |
| Tipos con marca | Alias `str` planos, forjables en cualquier lado | `NewType` para docs mas value objects frozen con `slots` creados solo por `parse`, construccion directa baneada por review | Inforjabilidad disciplinaria pese a runtime dinamico |
| Totalidad | `amount // parts` y `row["key"]` que lanzan o dan `None` en bordes | `Cents` mas `Result` fuerza manejo de cero, negativos y llaves faltantes con `mypy --strict` | Edge cases como obligacion de chequeo |
| Composicion | Piramides de `if` con `raise` en cada nivel | `and_then`, `map_result`, `map_err` y retorno temprano sobre el railway | Happy path lineal con riel de error tipado |
| Errores de dominio | `raise ValueError(str)`, atrapado como `Exception` | Union exhaustiva `DomainError`, `match` mas `assert_never` cubre cada variante | Nuevos casos de negocio rompen el chequeo de forma ruidosa |
| Errores de edge | Un solo `except Exception` que mapea todo a 400 | `AppError` con `cause`, shell mapea dominio a 4xx e infra a 500 con logs | Contexto rico donde humanos leen logs, tipos precisos donde el codigo ramifica |
| Estado de workflow | Flags como `is_paid` con `if` antes de cada accion | Type-state `OrderState[Draft]` a `OrderState[Paid]` con genericos | Transiciones ilegales son errores del checker |
| Costo en hot path | `model_validate` repetido en handler, servicio y repo | Parse una vez en el edge, pasa value objects con `slots` sin revalidar | Prueba sin impuesto de performance |
| Testing | Tests a mano con pocos literales | Hypothesis con cientos de inputs Unicode mas shrinking y `@example` | Confianza matematica en parsers, reproductores minimos |
| Arquitectura | Handlers mezclan Pydantic, DB y reglas con `async` en todos lados | Core puro sync con `calculate_refund` mas shell FastAPI y Pydantic delgado | Core testeable y portable, efectos aislados y auditables |

## 11. Reglas de oro y bibliografia

Primera regla: parsea una vez en el borde, nunca valides en el core.
`dict` y `str` crudos entran por HTTP o queues y se vuelven `Email`, `UserId` y `Cents` de inmediato.
El core solo acepta tipos probados y contiene cero `is_valid` y cero `model_validate` repetidos.

Segunda regla: haz estados ilegales irrepresentables y borra los guardias.
Prefiere uniones para alternativas, dataclasses frozen para combinaciones y value objects con smart constructors para invariantes.
Si una regla vive en un tipo, quita cada `if` que la rechequee aguas abajo.
Recuerda la salvedad Python: la privacidad es disciplinaria, asi que guarda la construccion directa con modulos y `mypy --strict`.

Tercera regla: escribe funciones totales y compon sobre el railway.
Retorna `Result` para cada operacion parcial, maneja cada variante y encadena con retorno temprano, `map_result` y `and_then`.
Reserva `raise` para bugs verdaderamente imposibles y bordes de infra, nunca para input de usuario.

Cuarta regla: estratifica errores por audiencia.
El dominio expone uniones exhaustivas `DomainError`.
La app envuelve fallas de infra una vez con `cause`.
El edge agrega contexto humano, logs y mapeo HTTP.
Nunca filtres `Exception` pelada desde APIs de dominio, y nunca dejes que el dominio importe el framework.

Quinta regla: mete workflows y costos al sistema de tipos.
Usa type-state con genericos para ciclos ordenados con dos o mas operaciones distintas.
Usa value objects con `slots` en hot paths en vez de reparsear con Pydantic.
Cubre parsers con Hypothesis y manten el shell FastAPI delgado alrededor de un core funcional puro.

Bibliografia del post.
Alexis King, `Parse, don't validate` (2019).
Paul Chiusano y Runar Bjarnason, `Functional Programming in Scala` (2014).
Harold Abelson y Gerald Jay Sussman, `Structure and Interpretation of Computer Programs` (1996).
Eric Evans, `Domain-Driven Design` (2003).
Edwin Brady, `Type-Driven Development with Idris` (2017).
Scott Wlaschin, `Railway Oriented Programming` (2013).
Gary Bernhardt, `Functional Core, Imperative Shell` (2012).
