---
name: error-handling
description: >
  Layers of Trust error handling architecture for this repo. Validate at edges
  with Result[T,E], parse with Pydantic + branded types, assert in core with
  assert_ok(). Never use bare assert.
  Use when writing new code, editing code, fixing bugs, or reviewing error/validation
  patterns. Load alongside coding-guide for code changes.
---

# Error Handling Architecture

Follow "Layers of Trust" pattern (see [REFERENCE.md](./REFERENCE.md) for full rationale and examples):

- **Validate at edges (defensive):** Use `Result[T, E] = Ok[T] | Err[E]` pattern for external input — API requests, DB reads, etc. Return user-friendly 4xx errors.
- **Parse, don't validate:** Use Pydantic + `NewType` branded types (`ValidatedUserId`, `PositiveAmount`) at the boundary to make invalid states unrepresentable. Core functions only accept branded types.
- **Assert in core (offensive):** Once inside trusted domain, use `assert_ok()` to fail-fast on invariant violations. Always a 5xx — page the developer.
- **Never use bare `assert`:** Python strips it under `-O`. Use custom assertion functions that always raise.
- `TypeGuard[Ok[T]]` with `is_ok()` permanently narrows types for the caller.