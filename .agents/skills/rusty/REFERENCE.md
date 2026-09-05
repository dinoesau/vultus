# Rusty - Referencia completa

Fuente: guia arquitectonica sobre error handling, invariantes y modelado funcional de dominio en Rust.
Cubre el post completo sin recortes: antipatron, cambio de paradigma y los 5 pilares mas ingenieria avanzada y arquitectura.
Lee solo la seccion que necesites para la tarea actual.

## Contenido

- [1. Antipatron del Rust defensivo](#1-antipatron-del-rust-defensivo)
- [2. Parse don't validate](#2-parse-dont-validate)
- [3. Pilar 1: newtypes y smart constructors](#3-pilar-1-newtypes-y-smart-constructors)
- [4. Pilar 2: ADTs y funciones totales](#4-pilar-2-adts-y-funciones-totales)
- [5. Pilar 3: Lisp, expresiones y metaprogramacion](#5-pilar-3-lisp-expresiones-y-metaprogramacion)
- [6. Pilar 4: errores estratificados](#6-pilar-4-errores-estratificados)
- [7. Pilar 5: type-state en compilacion](#7-pilar-5-type-state-en-compilacion)
- [8. Ingenieria avanzada: Ref zero-cost y proptest](#8-ingenieria-avanzada-ref-zero-cost-y-proptest)
- [9. Arquitectura: functional core, imperative shell](#9-arquitectura-functional-core-imperative-shell)
- [10. Tabla defensive vs type-driven](#10-tabla-defensive-vs-type-driven)
- [11. Reglas de oro y bibliografia](#11-reglas-de-oro-y-bibliografia)

## 1. Antipatron del Rust defensivo

El habito tentador es revisar cada input en cada funcion.
Ese habito es caro porque ninguna firma registra lo ya probado.
En Rust ese habito desperdicia el compilador.
El compilador no es un revisor de sintaxis.
El compilador es un probador de teoremas en tiempo de compilacion.

El ejemplo canonico es sopa de primitivos con `HashMap<String, i64>`.
`process_refund` recibe `user_id: String`, `email: String`, `amount_cents: i32`.
Cada funcion repite tres guardias: usuario no vacio, email con `@`, monto positivo.
`send_receipt` copia las mismas tres guardias.
Cada cambio a la regla de email exige edicion shotgun en varias capas.
El hot path paga escaneos y branches repetidos.
El acoplamiento crece porque reglas de identidad y dinero filtran a infraestructura.
El costo profundo es paranoia: como nada prueba lo ya validado, todo se rechequea.

```rust
use std::collections::HashMap;

pub struct Database {
    balances: HashMap<String, i64>,
}

impl Database {
    pub fn get_balance(&self, user_id: &str) -> Option<i64> {
        self.balances.get(user_id).copied()
    }
}

// Antipatron: primitivos fluyen por todas las capas.
pub fn process_refund(
    db: &Database,
    user_id: String,
    email: String,
    amount_cents: i32,
) -> Result<String, String> {
    if user_id.trim().is_empty() {
        return Err("invalid user_id".to_string());
    }
    if !email.contains('@') {
        return Err("invalid email".to_string());
    }
    if amount_cents <= 0 {
        return Err("invalid amount".to_string());
    }
    let balance = db
        .get_balance(&user_id)
        .ok_or_else(|| "user not found".to_string())?;
    if balance < amount_cents as i64 {
        return Err("insufficient funds".to_string());
    }
    Ok(format!("refunded {amount_cents} to {user_id}"))
}
```

El contrato que Rust permite es distinto.
Parsea datos no confiables una vez en el borde.
Entrega al core solo tipos que no pueden estar mal.
Borra los guardias duplicados para siempre.

## 2. Parse don't validate

Alexis King capturo la idea en `Parse, don't validate` (2019).
Validar inspecciona un valor y conserva el tipo debil.
Parsear consume el tipo debil y produce un tipo fuerte con la prueba incluida.
Esa distincion cambia la arquitectura.

Validar tiene esta forma.
Responde una pregunta y tira la respuesta a la basura.
Despues de `is_valid_email`, el valor sigue siendo `String`.
El compilador no aprendio nada.
La siguiente funcion debe revisar de nuevo.

```rust
pub fn is_valid_email(raw: &str) -> bool {
    let parts: Vec<&str> = raw.split('@').collect();
    parts.len() == 2 && !parts[0].is_empty() && parts[1].contains('.')
}

pub fn notify(raw_email: String) {
    if is_valid_email(&raw_email) {
        // raw_email sigue siendo String aqui.
        println!("sending to {raw_email}");
    }
}
```

Parsear tiene otra forma.
Transforma y certifica en un solo movimiento.
Despues de `Email::parse`, ninguna funcion aguas abajo rechequea el `@`.
El tipo es la prueba.
La firma `fn notify(email: Email)` documenta el invariante mejor que un comentario.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Email(String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmailError {
    MissingAt,
    EmptyLocalPart,
    InvalidDomain,
}

impl Email {
    pub fn parse(raw: String) -> Result<Self, EmailError> {
        let raw = raw.trim().to_string();
        let (_, domain) = raw.split_once('@').ok_or(EmailError::MissingAt)?;
        let local = raw.split('@').next().unwrap_or_default();
        if local.is_empty() {
            return Err(EmailError::EmptyLocalPart);
        }
        if !domain.contains('.') {
            return Err(EmailError::InvalidDomain);
        }
        Ok(Self(raw))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
```

La regla de borde es simple.
Los datos cruzan el borde como `String` e `i32`.
Viajan dentro del core como `Email`, `UserId` y `Cents`.
El parser vive en exactamente un modulo por tipo.
Todo lo que esta detras compone sin guardias.
Esto es el mismo movimiento que Edwin Brady ensena en `Type-Driven Development with Idris`.
Deja que el tipo guie el flujo.
Rechaza programas malos en compilacion en vez de descubrirlos en logs de produccion.

## 3. Pilar 1: newtypes y smart constructors

Eric Evans los llama Value Objects en `Domain-Driven Design`.
Son conceptos pequenos, inmutables y auto-validados sin identidad mas alla de su valor.
Rust los modela con el patron newtype mas smart constructor.
La privacidad del modulo hace la garantia fisica, no social.
El truco es el campo privado.

```rust
// src/domain/email.rs
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Email(String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmailError {
    MissingAt,
    EmptyLocalPart,
    InvalidDomain,
}

impl fmt::Display for EmailError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingAt => write!(f, "email must contain @"),
            Self::EmptyLocalPart => write!(f, "email local part is empty"),
            Self::InvalidDomain => write!(f, "email domain must contain ."),
        }
    }
}

impl std::error::Error for EmailError {}

impl Email {
    /// Smart constructor: la unica via para construir un Email.
    pub fn parse(raw: String) -> Result<Self, EmailError> {
        let trimmed = raw.trim().to_string();
        let (local, domain) = trimmed.split_once('@').ok_or(EmailError::MissingAt)?;
        if local.is_empty() {
            return Err(EmailError::EmptyLocalPart);
        }
        if !domain.contains('.') {
            return Err(EmailError::InvalidDomain);
        }
        Ok(Self(trimmed))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for Email {
    type Err = EmailError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s.to_string())
    }
}

impl fmt::Display for Email {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for Email {
    fn as_ref(&self) -> &str {
        &self.0
    }
}
```

Como el `String` interno es privado y el struct vive en su propio modulo, ningun otro modulo puede escribir `Email("garbage".to_string())`.
El compilador lo rechaza.
La unica via es `Email::parse`, que retorna `Result`.
Este es el borde de Aggregate de DDD, impuesto por `mod`, no por disciplina.
Aplica el mismo patron a cada primitivo con regla.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cents(u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoneyError {
    NonPositive,
}

impl Cents {
    pub fn parse(raw: i64) -> Result<Self, MoneyError> {
        if raw <= 0 {
            return Err(MoneyError::NonPositive);
        }
        Ok(Self(raw as u64))
    }

    /// Uso interno del crate solo despues de prueba previa.
    pub(crate) fn from_raw(value: u64) -> Self {
        Self(value)
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

impl TryFrom<i64> for Cents {
    type Error = MoneyError;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UserId(String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserIdError {
    Empty,
}

impl UserId {
    pub fn parse(raw: String) -> Result<Self, UserIdError> {
        let trimmed = raw.trim().to_string();
        if trimmed.is_empty() {
            return Err(UserIdError::Empty);
        }
        Ok(Self(trimmed))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
```

Ahora las firmas del core prueban sus precondiciones.
No hay chequeo de email en el core.
No hay chequeo de monto en el core.
Los tipos ya lo probaron.
Los accesores `as_str` y `Display` son intencionales.
Quien llama puede leer el valor pero no forjarlo.
Eso es encapsulacion sin costo de runtime.

## 4. Pilar 2: ADTs y funciones totales

Paul Chiusano y Runar Bjarnason ensenan esto en `Functional Programming in Scala`.
Modela con tipos precisos, escribe funciones totales y compon con combinadores.
Los enums y structs de Rust son tipos algebraicos de datos.
`Result` es tu monada `Either`.
Los sum types enumeran alternativas exclusivas.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardDetails {
    pub last_four: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferDetails {
    pub iban: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaymentMethod {
    Card(CardDetails),
    Transfer(TransferDetails),
    Cash,
}
```

Los product types combinan hechos independientes.
No hay null ni valor faltante implicito.
No hay campo de metodo como string sin estructura.
Un `match` sobre `PaymentMethod` debe cubrir cada brazo o el build falla.
Esa exhaustividad es una prueba sobre tus ramas de negocio.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Order {
    pub id: UserId,
    pub email: Email,
    pub amount: Cents,
    pub method: PaymentMethod,
}
```

Una funcion total esta definida para el 100 por ciento de sus inputs.
Nunca hace panic ni bloquea con input oculto.
Una funcion parcial finge ser total pero explota con algunos inputs.

```rust
// Parcial: panic con cero y overflow en debug.
pub fn refund_share_partial(amount: u64, parts: u64) -> u64 {
    amount / parts
}

// Total: cada input mapea a un resultado explicito.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitError {
    EmptyParts,
}

pub fn refund_share_total(amount: Cents, parts: u64) -> Result<Cents, SplitError> {
    if parts == 0 {
        return Err(SplitError::EmptyParts);
    }
    Ok(Cents::from_raw(amount.value() / parts))
}
```

La composicion usa `map`, `and_then` y `map_err` en vez de piramides de `if`.
Esto es Railway Oriented Programming de Scott Wlaschin, expresado con `Result`.
Hay dos rieles paralelos.
El riel feliz lleva valores hacia adelante.
El riel de error corta sin unwind.
El tipo de error dice al handler que status retornar.
Asi un typo 400 nunca se disfraza de outage 500.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderError {
    Email(EmailError),
    Money(MoneyError),
    User(UserIdError),
}

pub fn build_order(raw_email: String, raw_amount: i64) -> Result<Order, OrderError> {
    Email::parse(raw_email)
        .map_err(OrderError::Email)
        .and_then(|email| {
            Cents::parse(raw_amount)
                .map_err(OrderError::Money)
                .map(|amount| (email, amount))
        })
        .and_then(|(email, amount)| {
            UserId::parse("placeholder".to_string())
                .map_err(OrderError::User)
                .map(|id| Order {
                    id,
                    email,
                    amount,
                    method: PaymentMethod::Cash,
                })
        })
}
```

Mejor aun, usa el operador `?`, que es bind monadico con retorno temprano en el riel de error.
Ambas versiones conservan los dos rieles.
La version con `?` es lineal y mas legible.

```rust
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

## 5. Pilar 3: Lisp, expresiones y metaprogramacion

Rust hereda su alma de expresion de la tradicion Lisp, Scheme y ML que Abelson y Sussman celebran en `Structure and Interpretation of Computer Programs`.
Casi todo evalua a un valor.
Eso permite asignar resultados validados directo en vez de mutar temporales con cadenas de statements.
No hay danza de `let mut result`.
No hay variable sin inicializar.
El compilador revisa que cada rama produzca el tipo declarado.

```rust
pub fn classify(raw: &str) -> Result<Email, EmailError> {
    let email: Email = match Email::parse(raw.to_string()) {
        Ok(valid) => valid,
        Err(EmailError::MissingAt) => {
            // La ruta de reparacion tambien es expresion.
            Email::parse(format!("{raw}@example.com"))?
        }
        Err(other) => return Err(other),
    };
    Ok(email)
}

pub fn label(amount: Cents) -> &'static str {
    // if es expresion, no statement.
    if amount.value() > 1_000_00 {
        "enterprise"
    } else if amount.value() > 10_00 {
        "standard"
    } else {
        "micro"
    }
}
```

La idea Lisp profunda es codigo como dato.
SICP ensena a construir lenguajes embebidos donde programas manipulan programas.
Los macros procedurales de Rust hacen esto a nivel de arbol de sintaxis durante la compilacion.
Un macro lee tu definicion de struct como dato y emite el smart constructor, el tipo de error y los impls por ti.
El crate `nutype` es la version pragmatica de esa idea.
Declara el invariante y el macro genera el boilerplate.

```rust
use nutype::nutype;

#[nutype(validate(greater > 0), derive(Debug, Clone, Copy, PartialEq, Eq))]
pub struct CentsNutype(i64);

#[nutype(validate(not_empty, len_char_max = 254), derive(Debug, Clone, PartialEq, Eq))]
pub struct EmailNutype(String);
```

`nutype` expande a un newtype de campo privado con `try_from`, `FromStr`, `Display`, `AsRef` y un enum de error preciso.
Obtienes la misma garantia de privacidad de modulo que un smart constructor manual sin repetirlo veinte veces.
Cuando el invariante es especifico del dominio, escribe tu propio derive o attribute macro.
La forma siempre es igual.
Parsea el `TokenStream` de entrada a tipos `syn`, valida el AST y emite salida `quote` con el constructor.

```rust
// Sketch conceptual de un macro de smart constructor.
// Los proc macros reales viven en un crate -macros separado.
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

#[proc_macro_derive(NonEmptyString)]
pub fn non_empty_string(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let expanded = quote! {
        impl #name {
            pub fn parse(raw: String) -> Result<Self, crate::domain::NonEmptyError> {
                if raw.trim().is_empty() {
                    return Err(crate::domain::NonEmptyError::Empty);
                }
                Ok(Self(raw.trim().to_string()))
            }
        }
    };
    TokenStream::from(expanded)
}
```

Usa macros para quitar repeticion, nunca para esconder reglas de negocio.
El invariante debe seguir visible en la definicion del tipo.
El macro solo mecaniza la prueba.

## 6. Pilar 4: errores estratificados

No todos los errores pertenecen al mismo tipo.
Los errores de dominio son resultados de negocio esperados y deben ser exhaustivos.
Los errores de infraestructura son fallas operativas y necesitan contexto y backtraces.
Mezclarlos en un `String` o un error boxeado destruye esa senal.
Estratifica en tres capas.
El core expone errores de dominio con `thiserror`.
La app envuelve dominio mas infra una vez.
El edge reporta con `anyhow` y contexto.

Modela errores de dominio con `thiserror`.
Cada variante es un hecho de negocio que el llamador debe manejar.

```rust
// src/domain/error.rs
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DomainError {
    #[error("invalid email: {0}")]
    InvalidEmail(#[from] EmailError),
    #[error("invalid amount: must be positive")]
    InvalidAmount,
    #[error("user not found")]
    UserNotFound,
    #[error("insufficient funds")]
    InsufficientFunds,
    #[error("refund already processed for order {order_id}")]
    AlreadyRefunded { order_id: String },
}
```

El `match` exhaustivo ahora fuerza decisiones de producto.
Quitar una variante rompe este match en compilacion.
Esa ruptura es la funcionalidad buscada.

```rust
pub fn refund_status_code(err: &DomainError) -> u16 {
    match err {
        DomainError::InvalidEmail(_) | DomainError::InvalidAmount => 400,
        DomainError::UserNotFound => 404,
        DomainError::InsufficientFunds | DomainError::AlreadyRefunded { .. } => 422,
    }
}
```

Envuelve errores de infraestructura una vez en la capa de aplicacion.
No los mezcles con el dominio antes de tiempo.

```rust
// src/app/error.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("domain error: {0}")]
    Domain(#[from] DomainError),
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("payment gateway error: {0}")]
    Gateway(#[from] reqwest::Error),
}
```

Usa `anyhow` solo en el edge final: binarios, CLIs, scripts de migracion y el retorno de `main`.
Agrega contexto y backtraces donde humanos leen logs, no donde librerias definen contratos.

```rust
// src/main.rs
use anyhow::Context;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pool = sqlx::PgPool::connect(&std::env::var("DATABASE_URL")?)
        .await
        .context("failed to connect to Postgres")?;
    run_server(pool).await.context("server crashed")?;
    Ok(())
}
```

La regla es nitida.
Si publicas el tipo, usa `thiserror`.
Si corres el binario, usa `anyhow`.
Nunca retornes `anyhow::Error` desde funciones de dominio, porque borra los casos exhaustivos que tus llamadores necesitan.
Nunca uses errores `String` en codigo nuevo, porque borran estructura e impiden match programatico.

## 7. Pilar 5: type-state en compilacion

Algunos invariantes no son sobre valores solos sino sobre secuencias.
Una orden no se puede pagar antes de enviarse.
Un refund no se puede emitir dos veces.
Los booleanos de runtime como `is_submitted` se pueden olvidar o revisar en orden mal.
Type-state codifica el workflow en genericos para que secuencias malas no compilen.
Esto es diseno type-driven estilo Brady aplicado a ciclos de negocio.

```rust
// src/domain/order_state.rs
use std::marker::PhantomData;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Draft;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submitted;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Order<State> {
    id: String,
    amount: Cents,
    state: PhantomData<State>,
}

impl Order<Draft> {
    pub fn new(id: String, amount: Cents) -> Self {
        Self {
            id,
            amount,
            state: PhantomData,
        }
    }

    pub fn submit(self) -> Order<Submitted> {
        // self se mueve y se destruye aqui.
        // El valor Draft ya no se puede reusar.
        Order {
            id: self.id,
            amount: self.amount,
            state: PhantomData,
        }
    }
}

impl Order<Submitted> {
    pub fn pay(self) -> Order<Paid> {
        Order {
            id: self.id,
            amount: self.amount,
            state: PhantomData,
        }
    }
}

impl Order<Paid> {
    pub fn receipt(&self) -> String {
        format!("paid {} cents for {}", self.amount.value(), self.id)
    }
}
```

Ownership hace el trabajo pesado.
Cada transicion toma `self` por valor y retorna un estado nuevo.
El valor viejo se mueve y desaparece.
No hay use-after-free logico donde handles `Draft` rancios se reenvian.

```rust
let draft = Order::<Draft>::new("ord_1".to_string(), Cents::parse(5000).unwrap());
let submitted = draft.submit();
// draft.submit(); // Error de compilacion: valor movido.
let paid = submitted.pay();
println!("{}", paid.receipt());
// paid.pay(); // Error de compilacion: no hay metodo pay en Order<Paid>.
```

Usa type-state cuando la secuencia importa y el costo de una transicion mala es alto: pagos, provisioning, publicacion y onboarding multi-paso.
No lo uses para cada booleano, o el ruido generico ahoga el dominio.
Una buena heuristica es dos o mas estados ordenados con distintas operaciones disponibles.

## 8. Ingenieria avanzada: Ref zero-cost y proptest

Los newtypes owned como `Email(String)` alojan una vez y son perfectos para storage y APIs.
Los hot paths como routers, validadores y parsers deberian evitar hasta esa alocacion cuando solo prestan.
Los lifetimes permiten construir newtypes prestados con cero costo de heap.
`EmailRef<'a>` es un puntero mas una longitud.
Copia con `Copy`, nunca toca el allocator y aun garantiza el invariante.
Parsea prestado en el edge y promueve a owned solo cuando debas almacenar.
Esta es la promesa de abstraccion zero-cost: seguridad sin impuesto de runtime.

```rust
// Vista prestada zero-cost: sin alocacion, misma forma de prueba.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EmailRef<'a>(&'a str);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmailRefError {
    MissingAt,
    EmptyLocalPart,
    InvalidDomain,
}

impl<'a> EmailRef<'a> {
    pub fn parse(raw: &'a str) -> Result<Self, EmailRefError> {
        let trimmed = raw.trim();
        let (local, domain) = trimmed.split_once('@').ok_or(EmailRefError::MissingAt)?;
        if local.is_empty() {
            return Err(EmailRefError::EmptyLocalPart);
        }
        if !domain.contains('.') {
            return Err(EmailRefError::InvalidDomain);
        }
        Ok(Self(trimmed))
    }

    pub fn as_str(self) -> &'a str {
        self.0
    }

    pub fn to_owned_email(self) -> Email {
        // Unico punto de upgrade de prestado a owned.
        Email::parse(self.0.to_string()).expect("borrowed was already valid")
    }
}
```

La logica de parsing merece tests mas fuertes que ejemplos hechos a mano.
El testing basado en propiedades con `proptest` lanza miles de inputs sinteticos a tu smart constructor, incluyendo Unicode, caracteres de control y longitudes patologicas.
Corre con `cargo test`.
Si un caso falla, `proptest` lo reduce al reproductor minimo y guarda el seed.
Agrega ese seed como test de regresion.
Tu parser gana robustez matematica en vez de cobertura anecdotica.

```rust
// tests/email_properties.rs
use proptest::prelude::*;

proptest! {
    #[test]
    fn valid_emails_always_parse(s in "[a-z0-9]{1,16}@[a-z]{1,8}\\.[a-z]{2,4}") {
        prop_assert!(Email::parse(s).is_ok());
    }

    #[test]
    fn missing_at_never_parses(s in "[a-z0-9 ]{1,32}") {
        prop_assume!(!s.contains('@'));
        prop_assert!(Email::parse(s).is_err());
    }

    #[test]
    fn parse_never_panics(s in "\\PC*") {
        // Cualquier string Unicode debe mapear a Ok o Err, nunca panic.
        let _ = Email::parse(s);
    }

    #[test]
    fn trimmed_value_roundtrips(s in " *[a-z]{1,8}@example\\.com *") {
        let email = Email::parse(s.clone()).unwrap();
        prop_assert_eq!(email.as_str(), s.trim());
    }
}
```

## 9. Arquitectura: functional core, imperative shell

Gary Bernhardt resumo la arquitectura mas sana en una linea: Functional Core, Imperative Shell.
El core es puro, sincrono y total.
Toma tipos de dominio y retorna `Result`.
Sin `async`, sin sockets, sin estado global, sin lecturas de reloj.
El shell es delgado y con efectos.
Habla HTTP y JSON, parsea en el borde, llama al core y mapea errores tipados a status codes.
Define el core puro primero.

```rust
// src/core/refunds.rs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefundPolicy {
    pub max_cents: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refund {
    pub order_id: String,
    pub amount: Cents,
}

pub fn calculate_refund(
    order: &Order,
    requested: Cents,
    policy: &RefundPolicy,
) -> Result<Refund, DomainError> {
    // Funcion pura: sin IO, sin async, sin globales.
    // Todos los inputs ya son tipos probados.
    if requested.value() > order.amount.value() {
        return Err(DomainError::InsufficientFunds);
    }
    if requested.value() > policy.max_cents {
        return Err(DomainError::InvalidAmount);
    }
    Ok(Refund {
        order_id: order.id.as_str().to_string(),
        amount: requested,
    })
}
```

Los DTOs de Serde se quedan tontos y crudos en el shell.
No llevan invariantes ni logica.
Solo transportan lo que llego por la red.

```rust
// src/shell/dto.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct RefundRequestDto {
    pub order_id: String,
    pub email: String,
    pub amount_cents: i64,
}

#[derive(Debug, Serialize)]
pub struct RefundResponseDto {
    pub order_id: String,
    pub refunded_cents: u64,
}
```

El handler de Axum une los dos mundos y nada mas.
Parsea en el borde, rehidrata estado minimo, llama al core puro y mapea a DTO.
Sin logica de negocio aqui.

```rust
// src/shell/handlers.rs
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};

pub async fn refund_handler(
    State(policy): State<RefundPolicy>,
    Json(raw): Json<RefundRequestDto>,
) -> Result<impl IntoResponse, AppError> {
    // 1. Parsear en el borde: String e i64 se vuelven Email, UserId, Cents.
    let email = Email::parse(raw.email).map_err(DomainError::InvalidEmail)?;
    let user_id = UserId::parse(raw.order_id).map_err(|_| DomainError::UserNotFound)?;
    let amount = Cents::parse(raw.amount_cents).map_err(|_| DomainError::InvalidAmount)?;

    // 2. Rehidratar estado minimo y llamar al core puro.
    let order = Order {
        id: user_id,
        email,
        amount,
        method: PaymentMethod::Cash,
    };
    let refund = calculate_refund(&order, amount, &policy)?;

    // 3. Mapear a DTO. Sin logica de negocio aqui.
    let body = RefundResponseDto {
        order_id: refund.order_id,
        refunded_cents: refund.amount.value(),
    };
    Ok((StatusCode::OK, Json(body)))
}
```

Mapea errores tipados a HTTP una vez, de forma exhaustiva.
Los errores de dominio se vuelven 400, 404 o 422 segun el caso.
Los errores de infra se vuelven 500 generico sin filtrar detalles.

```rust
impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self {
            Self::Domain(err) => match err {
                DomainError::InvalidEmail(_) | DomainError::InvalidAmount => {
                    (StatusCode::BAD_REQUEST, err.to_string())
                }
                DomainError::UserNotFound => (StatusCode::NOT_FOUND, err.to_string()),
                DomainError::InsufficientFunds | DomainError::AlreadyRefunded { .. } => {
                    (StatusCode::UNPROCESSABLE_ENTITY, err.to_string())
                }
            },
            Self::Database(_) | Self::Gateway(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal error".to_string(),
            ),
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}
```

El testing se divide limpio.
Prueba `calculate_refund` con structs planos y sin mocks.
Prueba el handler con `tower::ServiceExt::oneshot` y payloads JSON reales.
El core queda rapido y determinista porque los efectos viven solo en el shell.

## 10. Tabla defensive vs type-driven

Guarda esta tabla como checklist de review.
Si una fila se mueve a la izquierda, regresa la prueba al tipo.

| Concepto | Defensive Rust | Type-Driven Rust | Beneficio |
|---|---|---|---|
| Parseo de borde | `if` repetidos en cada funcion sobre `String` crudo | `Email::parse(String)` retorna `Result<Email, EmailError>` una vez | Una sola fuente de verdad, cero rechequeos en el core |
| Newtypes | Alias de `String` forjables en cualquier lado | `pub struct Email(String)` con campo privado, inforjable fuera de `mod` | Imposibilidad fisica de construccion invalida |
| Totalidad | Division e indexado que hacen panic en bordes | `Cents(u64)` mas `Result` fuerza manejo de cero y negativos | Edge cases como obligacion de compilacion |
| Composicion | Piramides de `if` con `return Err` en cada nivel | `and_then`, `map`, `map_err` y `?` sobre el railway de `Result` | Happy path lineal con riel de error tipado |
| Errores de dominio | `Result<T, String>` facil de tragar o mal clasificar | Enum exhaustivo con `thiserror`, `match` cubre cada variante | Nuevos casos de negocio rompen el build de forma ruidosa |
| Errores de edge | Un solo tipo catch-all hasta `main` | `anyhow` con `.context()` solo en `main` y binarios | Contexto rico donde humanos leen logs, tipos precisos donde el codigo ramifica |
| Estado de workflow | Flags como `is_paid` con `if` antes de cada accion | Type-state `Order<Draft>` a `Order<Paid>` con move semantics | Transiciones ilegales no compilan, handles rancios se destruyen |
| Costo en hot path | Newtypes `String` clonados en cada capa | `EmailRef<'a>` prestado sin heap, promote a owned una vez | Prueba sin impuesto de performance |
| Testing | `#[test]` hechos a mano con pocos literales | `proptest` con miles de inputs Unicode mas shrinking | Confianza matematica en parsers, reproductores minimos |
| Arquitectura | Handlers mezclan Serde, DB y reglas con `async` en todos lados | Core puro sync con `calculate_refund` mas shell Axum y Serde delgado | Core testeable y portable, efectos aislados y auditables |

## 11. Reglas de oro y bibliografia

Primera regla: parsea una vez en el borde, nunca valides en el core.
`String` e `i32` crudos entran por Serde o CLI y se vuelven `Email`, `UserId` y `Cents` de inmediato.
Las funciones del core solo aceptan tipos probados y contienen cero chequeos `is_valid`.

Segunda regla: haz estados ilegales irrepresentables y borra los guardias.
Prefiere `enum` para alternativas, `struct` para combinaciones y newtypes de campo privado para invariantes.
Si una regla vive en un tipo, quita cada `if` que la rechequee aguas abajo.

Tercera regla: escribe funciones totales y compon sobre el railway.
Retorna `Result` para cada operacion parcial, maneja cada variante y encadena con `?`, `map` y `and_then`.
Reserva panic para bugs verdaderamente imposibles, nunca para input de usuario.

Cuarta regla: estratifica errores por audiencia.
Las librerias de dominio exponen enums exhaustivos con `thiserror`.
Las apps envuelven fallas de infra una vez.
Los binarios agregan contexto humano con `anyhow`.
Nunca filtres `anyhow` ni errores `String` desde APIs de dominio.

Quinta regla: mete workflows y costos al sistema de tipos.
Usa type-state con move semantics para ciclos ordenados.
Usa vistas prestadas `EmailRef<'a>` en hot paths.
Cubre parsers con `proptest` y manten el shell Axum delgado alrededor de un core funcional puro.
Deja de defender cada funcion contra datos ya revisados.
Prueba una vez, codifica en un tipo y deja que `rustc` monte guardia mientras modelas el dominio.

Bibliografia completa del post.
Alexis King, `Parse, don't validate` (2019).
Paul Chiusano y Runar Bjarnason, `Functional Programming in Scala` (2014).
Harold Abelson y Gerald Jay Sussman, `Structure and Interpretation of Computer Programs` (1996).
Eric Evans, `Domain-Driven Design` (2003).
Edwin Brady, `Type-Driven Development with Idris` (2017).
Scott Wlaschin, `Railway Oriented Programming` (2013).
Gary Bernhardt, `Functional Core, Imperative Shell` (2012).
