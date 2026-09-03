# Layers of Trust: Error Handling Architecture — Reference

> *"A junior developer writes code that assumes everything will go right. A mid-level developer writes code that assumes everything will go wrong. A senior developer uses the type system to make going wrong impossible, deleting 80% of those checks entirely."*

---

## Core Distinction: Validation vs Assertion

| Feature | Validation | Assertion |
| :--- | :--- | :--- |
| **Location** | The Front Door (API/IO Boundary) | The VIP Room (Core Logic) |
| **Expectation** | Bad data is *expected* | Data is *trusted* |
| **Philosophy** | Defensive (Bouncer) | Offensive (Security Guard) |
| **Outcome** | Graceful Recovery → 4xx | Immediate Crash → 5xx |

- **Validation** inspects untrusted external input and recovers gracefully.
- **Assertion** inspects internal logic and fails fast when invariants break.
- Mixing them makes systems fragile: crashing on bad user input = bad UX; silently recovering from impossible internal state = data corruption.

---

## The Three Rules

### Rule 1: At the Edge, Errors are Values (`Result[T, E]`)

External data (API input, DB reads) is untrusted. `raise` is a hidden GOTO — the `except` block loses type info. Instead, return errors as values:

```python
@dataclass(frozen=True)
class Ok[T]:
    value: T

@dataclass(frozen=True)
class Err[E]:
    error: E

type Result[T, E] = Ok[T] | Err[E]


async def parse_json(req: Request) -> Result[dict, str]:
    try:
        return Ok(await req.json())
    except Exception:
        return Err("Malformed JSON")
```

Usage with `match` (Python 3.10+):

```python
match await parse_json(req):
    case Ok(body):
        ...   # body is typed as `dict`
    case Err(error):
        return JSONResponse({"error": error}, status_code=400)
```

### Rule 2: Parse, Don't Validate — Branded Types (`NewType`)

Traditional validation (`is_valid_email() -> bool`) doesn't leave a trace — downstream code still sees a raw `str` and re-validates defensively. The fix: **transform and brand**.

```python
from typing import NewType
from pydantic import BaseModel, field_validator, ValidationError

# Zero-cost at runtime, exists only for the type checker
ValidatedUserId = NewType("ValidatedUserId", str)
PositiveAmount = NewType("PositiveAmount", float)


class _RefundInput(BaseModel):
    user_id: str
    amount: float

    @field_validator("user_id")
    @classmethod
    def _must_be_uuid(cls, v: str) -> str:
        uuid.UUID(v)
        return v

    @field_validator("amount")
    @classmethod
    def _must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v


@dataclass(frozen=True)
class TrustedRefund:
    user_id: ValidatedUserId
    amount: PositiveAmount


def parse_refund_request(data: dict) -> Result[TrustedRefund, list[dict]]:
    try:
        raw = _RefundInput.model_validate(data)
        return Ok(TrustedRefund(
            user_id=ValidatedUserId(raw.user_id),  # Brand it
            amount=PositiveAmount(raw.amount),      # Brand it
        ))
    except ValidationError as e:
        return Err(e.errors(include_input=False))
```

After `parse_refund_request`, the type is `TrustedRefund.user_id: ValidatedUserId` — **not** `str`. Any core function accepting `ValidatedUserId` is mathematically guaranteed to only receive validated data. The type checker rejects raw strings.

### Rule 3: Inside the Core, Assert and Crash (`assert_ok`)

Once inside the trusted domain, invariants must hold. If they don't, it's a bug — crash immediately.

```python
from typing import TypeGuard, TypeVar

T = TypeVar("T")
E = TypeVar("E")


def is_ok(result: Result[T, E]) -> TypeGuard[Ok[T]]:
    """Type-narrowing predicate: after `if is_ok(x)`, x is `Ok[T]`."""
    return isinstance(result, Ok)


class InternalError(Exception):
    """System invariant violation. Always a 500."""


def assert_ok(result: Result[T, E]) -> Ok[T]:
    """Fail-fast assertion. Not disabled by -O (unlike bare `assert`)."""
    if not isinstance(result, Ok):
        raise InternalError(str(result.error))
    return result
```

**Never use bare `assert`:** Python strips it under `-O`. Always use a custom function that raises unconditionally.

---

## Full Architecture: Layers of Trust

```python
# --- Layer 0: Infrastructure ---
T = TypeVar("T"); E = TypeVar("E")

def is_ok(result: Result[T, E]) -> TypeGuard[Ok[T]]:
    return isinstance(result, Ok)

def assert_ok(result: Result[T, E]) -> Ok[T]:
    if not isinstance(result, Ok):
        raise InternalError(str(result.error))
    return result


# --- Layer 1: Core Domain (ZERO validation checks) ---
# Because it demands Branded Types, dirty data is impossible.

async def execute_refund(user_id: ValidatedUserId, amount: PositiveAmount) -> None:
    db_result = await get_user(user_id)       # DB reads are "edges" too → Result
    ok_user = assert_ok(db_result)            # Fail-fast if invariant broken
    await gateway.refund(ok_user.value.stripe_id, amount)


# --- Layer 2: Edge Controller ---

@app.post("/refund")
async def process_refund_route(req: Request):
    # Step 1: Parse the chaotic outside world
    json_result = await parse_json(req)
    if not is_ok(json_result):
        return JSONResponse({"error": json_result.error}, status_code=400)

    # Step 2: Smart Constructor (Validate + Brand)
    payload_result = parse_refund_request(json_result.value)
    if not is_ok(payload_result):
        return JSONResponse({"error": payload_result.error}, status_code=400)

    # Step 3: Execute core logic
    try:
        refund = payload_result.value
        await execute_refund(refund.user_id, refund.amount)
        return JSONResponse({"message": "Success"}, status_code=200)
    except InternalError as error:
        log_critical_bug(error)  # Page the developer
        return JSONResponse({"error": "Internal Server Error"}, status_code=500)
```

Notice `execute_refund` has **zero** `if` checks, `isinstance` guards, or defensive validation. Cognitive load drops to zero.

---

## Anti-Patterns to Avoid

| Anti-Pattern | Problem | Fix |
| :--- | :--- | :--- |
| `return None` on error | Loses the `E` — caller can't distinguish error types | Return `Err[E]` |
| Generic `try/except Exception` swallowing everything | Can't tell 4xx from 5xx | Only catch at edges, use `Result` |
| Bare `assert` | Stripped under `python -O` | Use `assert_ok()` |
| Validating the same data in multiple layers | No "receipt" that validation happened | Brand with `NewType`, trust the type |
| `if not data:` checks in core logic | Defensive clutter, hides bugs | Let `assert_ok` catch invariant violations |

---

## Summary

1. **Validate at edges (defensive):** `Result[T, E]` for external input → 4xx.
2. **Parse, don't validate:** Pydantic + `NewType` branded types at the boundary.
3. **Assert in core (offensive):** `assert_ok()` for invariants → 5xx.
4. **Never bare `assert`:** stripped under `-O`.
5. **`TypeGuard[Ok[T]]` + `is_ok()`** permanently narrows types for the caller.
