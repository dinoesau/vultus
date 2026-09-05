use super::error::{CoreError, Result};

/// Fails fast on broken invariants inside trusted core.
/// Never use bare `assert` (stripped under optimisations in some runtimes;
/// in Rust debug/release semantics differ). Always a 5xx: pages the developer.
pub fn assert_ok(cond: bool, msg: &'static str) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(CoreError::Invariant(msg))
    }
}
