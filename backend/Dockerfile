FROM rust:1.94-slim AS builder
WORKDIR /work
COPY Cargo.toml Cargo.toml
COPY Cargo.lock Cargo.lock
COPY crates crates
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN cargo build --release -p vultus-api

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /work/target/release/vultus-api /usr/local/bin/vultus-api
ENV RUST_LOG=info PORT=8000 QUEUE_DRIVER=memory R2_TTL_SECONDS=60
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["vultus-api"]
