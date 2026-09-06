# syntax=docker/dockerfile:1
# Etapa chef: calcula la receta de dependencias con cargo-chef para que los
# cambios en `crates/` no invaliden la capa de deps (la parte lenta).
FROM rust:1.94-slim AS chef
WORKDIR /work
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN cargo install cargo-chef --locked
COPY Cargo.toml Cargo.toml
COPY Cargo.lock Cargo.lock
COPY crates crates
RUN cargo chef prepare --recipe-path recipe.json

FROM rust:1.94-slim AS builder
WORKDIR /work
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY --from=chef /usr/local/cargo/bin/cargo-chef /usr/local/cargo/bin/cargo-chef
COPY --from=chef /work/recipe.json recipe.json
# Precocina solo dependencias: cachea el paso lento entre cambios de codigo.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/work/target \
    cargo chef cook --release --recipe-path recipe.json
COPY Cargo.toml Cargo.toml
COPY Cargo.lock Cargo.lock
COPY crates crates
# El binario vive en el mount de target: copiarlo fuera para la etapa runtime.
RUN --mount=type=cache,target=/work/target \
    cargo build --release --locked -p vultus-api && cp /work/target/release/vultus-api /tmp/vultus-api

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /tmp/vultus-api /usr/local/bin/vultus-api
ENV RUST_LOG=info PORT=8000 QUEUE_DRIVER=memory R2_TTL_SECONDS=60
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["vultus-api"]
