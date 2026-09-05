---
name: rusty
description: >
  Diseña dominios Rust con tipos que prueban invariantes en compilación.
  Usa cuando haya validación repetida con if sobre String o i32, firmas con primitivos,
  errores String o anyhow en el dominio, o se necesiten newtype con smart constructor,
  parse-don-t-validate, ADTs y funciones totales, thiserror vs anyhow, type-state,
  EmailRef zero-cost, proptest, nutype, Axum con functional-core-imperative-shell.
  No usar para Python o TypeScript, solo Rust.
license: MIT
allowed-tools: Bash
---

# Rusty - Modelado de dominio funcional en Rust

Convierte validacion repetida en runtime en pruebas hechas por el compilador.
Parsea una vez en el borde y deja que el core componga tipos que ya son correctos.

## Cuando usar esta skill

Usa esta skill cuando el codigo Rust muestre estos sintomas.
Repite los mismos `if` sobre `String` o `i32` en varias capas.
Las firmas reciben primitivos y nadie sabe que ya fue probado.
Los errores son `String` y no se pueden clasificar por programa.
El dominio usa `anyhow` y pierde exhaustividad.
Hay booleanos como `is_paid` que se revisan en orden manual.
El hot path clona `String` solo para validar de nuevo.

No uses esta skill para Python o TypeScript.
No la uses para logica ya modelada con tipos probados.
No la uses para decisiones de infraestructura sin invariantes de dominio.

## Workflow

Copia este checklist en tu respuesta y marca avance.

```
Progreso Rusty:
- [ ] 1. Parsear en el borde a tipos probados
- [ ] 2. Modelar con newtype y campo privado
- [ ] 3. Totalizar con Result y railway
- [ ] 4. Estratificar errores por audiencia
- [ ] 5. Verificar con compilador y tests
```

### 1. Parsear en el borde, nunca validar en el core

Convierte `String` y `i64` en `Email`, `UserId` y `Cents` en cuanto cruzan Serde, CLI o query params.
El core solo acepta tipos probados.
Si una regla vive en un tipo, borra todo `if` que la rechequee aguas abajo.
Ver [REFERENCE.md](./REFERENCE.md#2-parse-dont-validate).

### 2. Modelar con newtype y campo privado

Crea un modulo por concepto con `pub struct Email(String)` y campo privado.
Expón un unico smart constructor `parse` que retorna `Result`.
Agrega accesores de lectura (`as_str`, `Display`, `AsRef`) sin via de forja.
Ver [REFERENCE.md](./REFERENCE.md#3-pilar-1-newtypes-y-smart-constructors).

Plantilla estricta (baja libertad, seguir literal):

```rust
pub struct Email(String);

impl Email {
    pub fn parse(raw: String) -> Result<Self, EmailError> {
        // Validar aqui una sola vez.
        // Retornar Ok(Self(valor_normalizado)).
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
```

### 3. Totalizar con ADTs y railway

Modela alternativas con `enum` y combinaciones con `struct`.
Haz cada funcion parcial una funcion total que retorna `Result`.
Compon con `?`, `map`, `map_err` y `and_then` en vez de piramides de `if`.
Ver [REFERENCE.md](./REFERENCE.md#4-pilar-2-adts-y-funciones-totales).

### 4. Estratificar errores por audiencia

Si publicas el tipo, usa `thiserror` con enum exhaustivo.
Si corres el binario, usa `anyhow` con `.context()` solo en el edge.
Envuelve infraestructura una vez en `AppError`.
Nunca retornes `anyhow::Error` ni `String` desde el dominio.
Ver [REFERENCE.md](./REFERENCE.md#6-pilar-4-errores-estratificados).

### 5. Verificar antes de entregar

Ejecuta este loop y repite hasta que pase todo.
Si algo falla, corrige y vuelve a correr desde el paso 1.

```bash
cargo check
cargo clippy -- -D warnings
cargo test
```

Busca fugas del patron con estos greps.
Si alguno imprime lineas en `src/domain` o `src/core`, corrige antes de entregar.

```bash
grep -rn 'Result<.*, String>' src/domain src/core || true
grep -rn 'pub struct .* (pub ' src/domain || true
grep -rn '\.unwrap()' src/domain src/core || true
grep -rn 'anyhow' src/domain src/core || true
```

## Decisiones condicionales

Si creas contenido nuevo, sigue el workflow completo en orden.
Si editas codigo existente, empieza por el paso 1 solo en el borde tocado.
Si el error es de negocio esperado, agregalo como variante `thiserror`.
Si el error es operativo con causa externa, envuelvelo en `AppError`.
Si el workflow tiene dos o mas estados ordenados con distintas operaciones, usa type-state.
Si es un solo booleano sin orden, no uses type-state.
Si el hot path solo presta el valor, usa la vista prestada `EmailRef`.
Si debes almacenar el valor, promueve a owned una sola vez.

## Tabla anti-racionalizacion

| Excusa | Realidad |
|---|---|
| "Es solo un `String` rapido" | Ese atajo crea la proxima edicion shotgun en tres capas |
| "`anyhow` en todos lados es mas simple" | Borra los casos exhaustivos que el llamador necesita |
| "Un `bool is_paid` basta" | El orden se puede olvidar; el type-state no compila mal |
| "`unwrap` aqui nunca falla" | Si es input de usuario, retorna `Result`; reserva panic para bugs imposibles |
| "Clono el `String` por claridad" | En hot path usa `EmailRef` y promueve una vez |

## Referencias de un nivel

Lee solo lo necesario para la tarea actual.
No sigas links anidados mas alla de estos archivos.

- Patron completo y codigo: [REFERENCE.md](./REFERENCE.md).
- Pares before/after copiables: [EXAMPLES.md](./EXAMPLES.md).
