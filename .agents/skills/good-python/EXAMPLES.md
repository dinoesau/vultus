# Good-Python - Ejemplos before/after

Cada ejemplo muestra el patron defensivo y su reemplazo type-driven.
Copia el lado derecho como punto de partida.
Todos los snippets corren con las definiciones de [REFERENCE.md](./REFERENCE.md).

## 1. Validar en todos lados vs parsear una vez

Before: mismos tres guardias copiados en cada funcion.
After: firmas que prueban sus precondiciones.

```python
# Before: paranoia repetida.
def send_receipt(user_id: str, email: str, amount: int) -> None:
    if user_id.strip() == "":
        raise ValueError("invalid user_id")
    if "@" not in email:
        raise ValueError("invalid email")
    if amount <= 0:
        raise ValueError("invalid amount")

# After: el tipo ya probo todo.
def send_receipt_typed(user_id: UserId, email: Email, amount: Cents) -> None:
    _ = (user_id, email, amount)
```

## 2. `is_valid` booleano vs smart constructor

Before: la respuesta se tira y el tipo sigue debil.
After: el valor sale certificado del borde.

```python
# Before: el checker no aprende nada.
def notify(raw_email: str) -> None:
    if is_valid_email(raw_email):
        print(f"sending to {raw_email}")

# After: downstream ya no rechequea el @.
def notify_parsed(email: Email) -> None:
    print(f"sending to {email}")
```

## 3. `NewType` forjable vs value object

Before: una llamada forja el invariante.
After: solo `parse` crea el valor.

```python
# Before: se forja instantaneo.
charge(ValidatedUserId(""), PositiveAmount(-99.0))

# After: el unico camino retorna Result.
result = Cents.parse(raw_amount)
if isinstance(result, Err):
    return
apply_charge(result.value)
```

## 4. Division parcial vs total

Before: explota con cero.
After: el borde queda explicito en la firma.

```python
# Before.
def refund_share_partial(amount: int, parts: int) -> int:
    return amount // parts

# After.
def refund_share_total(amount: Cents, parts: int) -> Result[Cents, SplitError]:
    if parts <= 0:
        return Err(EmptyParts())
    raw = amount.to_int()
    if raw % parts != 0:
        return Err(NotDivisible(amount=raw, parts=parts))
    return Ok(Cents(_value=raw // parts))
```

## 5. Piramide de `if` vs railway con retorno temprano

Before: niveles anidados por cada parseo.
After: happy path lineal con riel de error tipado.

```python
# After: version recomendada.
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

## 6. `str` de error vs `DomainError` exhaustivo

Before: imposible clasificar por programa.
After: cada variante mapea a un status distinto.

```python
# Before.
def process(raw: str) -> str:
    if raw == "":
        raise ValueError("something failed")
    return raw

# After.
def classify_failure(err: DomainError) -> int:
    return domain_to_status(err)  # 400, 404 o 422 segun variante.
```

## 7. Booleanos de estado vs type-state

Before: el orden se puede olvidar.
After: el orden malo lo rechaza el checker.

```python
# Before: flag manual.
def pay_if_submitted(order_id: str, is_submitted: bool) -> str:
    if not is_submitted:
        raise ValueError("not submitted")
    return f"paid {order_id}"

# After: solo OrderState[Submitted] entra.
def pay_order(order: OrderState[Submitted]) -> OrderState[Paid]:
    return OrderState(order_id=order.order_id, amount=order.amount, state=Paid())
```

## 8. `model_validate` repetido vs parse-once

Before: el hot path paga el parseo tres veces.
After: el value object viaja probado sin revalidar.

```python
# Before: redundante.
def charge_twice(raw_amount: object) -> None:
    first = Cents.parse(raw_amount)
    if isinstance(first, Err):
        return
    second = Cents.parse(first.value.to_int())
    if isinstance(second, Err):
        return

# After: una sola prueba.
def charge_once(raw_amount: object) -> None:
    parsed = Cents.parse(raw_amount)
    if isinstance(parsed, Err):
        return
    apply_charge(parsed.value)
```
