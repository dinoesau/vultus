# Rusty - Ejemplos before/after

Cada ejemplo muestra el patron defensivo y su reemplazo type-driven.
Copia el lado derecho como punto de partida.
Todos los snippets compilan con las definiciones de [REFERENCE.md](./REFERENCE.md).

## 1. Validar en todos lados vs parsear una vez

Before: mismos tres guardias copiados en cada funcion.
After: firmas que prueban sus precondiciones.

```rust
// Before: paranoia repetida.
pub fn send_receipt(user_id: String, email: String, amount_cents: i32) -> Result<(), String> {
    if user_id.trim().is_empty() {
        return Err("invalid user_id".to_string());
    }
    if !email.contains('@') {
        return Err("invalid email".to_string());
    }
    if amount_cents <= 0 {
        return Err("invalid amount".to_string());
    }
    Ok(())
}

// After: el tipo ya probo todo.
pub fn send_receipt_typed(user_id: UserId, email: Email, amount: Cents) -> Result<(), DomainError> {
    let _ = (user_id, email, amount);
    Ok(())
}
```

## 2. `is_valid` booleano vs smart constructor

Before: la respuesta se tira y el tipo sigue debil.
After: el valor sale certificado del borde.

```rust
// Before: el compilador no aprende nada.
pub fn notify_raw(raw_email: String) {
    if is_valid_email(&raw_email) {
        println!("sending to {raw_email}");
    }
}

// After: downstream ya no rechequea el @.
pub fn notify_typed(email: Email) {
    println!("sending to {}", email.as_str());
}
```

## 3. Division parcial vs total

Before: explota con cero.
After: el borde queda explicito en la firma.

```rust
// Before.
pub fn refund_share_partial(amount: u64, parts: u64) -> u64 {
    amount / parts
}

// After.
pub fn refund_share_total(amount: Cents, parts: u64) -> Result<Cents, SplitError> {
    if parts == 0 {
        return Err(SplitError::EmptyParts);
    }
    Ok(Cents::from_raw(amount.value() / parts))
}
```

## 4. Piramide de `if` vs railway con `?`

Before: niveles anidados por cada parseo.
After: happy path lineal con riel de error tipado.

```rust
// After: version recomendada.
pub fn build_order_clean(raw_email: String, raw_amount: i64) -> Result<Order, OrderError> {
    let email = Email::parse(raw_email).map_err(OrderError::Email)?;
    let amount = Cents::parse(raw_amount).map_err(OrderError::Money)?;
    let id = UserId::parse("placeholder".to_string()).map_err(OrderError::User)?;
    Ok(Order {
        id,
        email,
        amount,
        method: PaymentMethod::Cash,
    })
}
```

## 5. `String` de error vs `thiserror` exhaustivo

Before: imposible clasificar por programa.
After: cada variante mapea a un status distinto.

```rust
// Before.
pub fn process(raw: String) -> Result<String, String> {
    Err("something failed".to_string())
}

// After.
pub fn classify_failure(err: &DomainError) -> u16 {
    match err {
        DomainError::InvalidEmail(_) | DomainError::InvalidAmount => 400,
        DomainError::UserNotFound => 404,
        DomainError::InsufficientFunds | DomainError::AlreadyRefunded { .. } => 422,
    }
}
```

## 6. Booleanos de estado vs type-state

Before: el orden se puede olvidar.
After: el orden malo no compila.

```rust
// Before: flag manual.
pub struct OrderFlag {
    pub paid: bool,
}

// After: transicion por movimiento.
let draft = Order::<Draft>::new("ord_1".to_string(), Cents::parse(5000).unwrap());
let submitted = draft.submit();
let paid = submitted.pay();
println!("{}", paid.receipt());
```

## 7. Clon en hot path vs vista prestada

Before: aloca solo para validar de nuevo.
After: prueba sin heap y promueve una vez.

```rust
// After: parse prestado en router, owned solo para guardar.
let view = EmailRef::parse("user@example.com")?;
let owned: Email = view.to_owned_email();
```

## 8. Core puro vs handler delgado

El core no toca IO ni `async`.
El shell parsea, llama y mapea.

```rust
// Core: testeable sin mocks.
let refund = calculate_refund(&order, requested, &policy)?;

// Shell: parsea en el borde y retorna DTO.
let email = Email::parse(raw.email).map_err(DomainError::InvalidEmail)?;
let amount = Cents::parse(raw.amount_cents).map_err(|_| DomainError::InvalidAmount)?;
```
