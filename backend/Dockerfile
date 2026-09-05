FROM rust:1.94-slim AS builder
WORKDIR /work
COPY Cargo.toml Cargo.toml
COPY crates crates
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN cargo build --release -p vultus-api

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /work/target/release/vultus-api /usr/local/bin/vultus-api
ENV RUST_LOG=info ML_SIDECAR_URL=http://ml-sidecar:8081
EXPOSE 8000
CMD ["vultus-api"]
