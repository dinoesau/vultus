use thiserror::Error;
use vultus_core::TtlSecs;

/// Error taxonomico de arranque: solo env -> tipos probados.
/// Exhaustivo para que `main` agregue contexto una vez via `anyhow`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConfigError {
    #[error("invalid PORT: {0}")]
    InvalidPort(String),
    #[error("invalid QUEUE_DRIVER: {0}")]
    InvalidDriver(String),
    #[error("invalid R2_TTL_SECONDS: {0}")]
    InvalidTtl(String),
}

/// Puerto TCP ya probado: 1..=65535.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Port(u16);

impl Port {
    pub fn parse(raw: &str) -> Result<Self, ConfigError> {
        let v: u16 = raw
            .trim()
            .parse()
            .map_err(|_| ConfigError::InvalidPort(raw.to_string()))?;
        if v == 0 {
            return Err(ConfigError::InvalidPort(raw.to_string()));
        }
        Ok(Self(v))
    }

    pub fn value(self) -> u16 {
        self.0
    }
}

impl Default for Port {
    fn default() -> Self {
        Self(8000)
    }
}

impl std::fmt::Display for Port {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Driver de queue: local en memoria o puntero R2 (paridad prod).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueueDriver {
    #[default]
    Memory,
    R2Pointer,
}

impl QueueDriver {
    pub fn parse(raw: &str) -> Result<Self, ConfigError> {
        match raw.trim().to_lowercase().as_str() {
            "" | "memory" => Ok(Self::Memory),
            "r2pointer" | "r2_pointer" | "cloudflare" | "r2" => Ok(Self::R2Pointer),
            other => Err(ConfigError::InvalidDriver(other.to_string())),
        }
    }
}

/// Config ya probada desde env. Todo con default para `cargo run` sin env.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    pub port: Port,
    pub driver: QueueDriver,
    pub ttl: TtlSecs,
}

impl Config {
    pub fn parse_env() -> Result<Self, ConfigError> {
        let port = std::env::var("PORT")
            .ok()
            .map(|v| Port::parse(&v))
            .transpose()?
            .unwrap_or_default();
        let driver = std::env::var("QUEUE_DRIVER")
            .ok()
            .map(|v| QueueDriver::parse(&v))
            .transpose()?
            .unwrap_or_default();
        let ttl = std::env::var("R2_TTL_SECONDS")
            .or_else(|_| std::env::var("RESULT_TTL_SECONDS"))
            .ok()
            .map(|v| {
                v.trim()
                    .parse::<u64>()
                    .map_err(|_| ConfigError::InvalidTtl(v.clone()))
                    .and_then(|n| TtlSecs::parse(n).map_err(|_| ConfigError::InvalidTtl(v.clone())))
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self { port, driver, ttl })
    }

    pub fn bind_addr(self) -> String {
        format!("0.0.0.0:{}", self.port.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_port_rejects_zero_and_garbage() {
        assert!(Port::parse("0").is_err());
        assert!(Port::parse("abc").is_err());
        assert_eq!(Port::parse("8000").expect("p").value(), 8000);
    }

    #[test]
    fn test_driver_parses_aliases() {
        assert_eq!(
            QueueDriver::parse("memory").expect("m"),
            QueueDriver::Memory
        );
        assert_eq!(
            QueueDriver::parse("cloudflare").expect("c"),
            QueueDriver::R2Pointer
        );
        assert!(QueueDriver::parse("redis").is_err());
    }
}
