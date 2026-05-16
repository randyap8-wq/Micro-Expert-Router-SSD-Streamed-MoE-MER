# --------------------------------------------------------------------
# micro-expert-router — SSD-streamed MoE engine
#
# Multi-stage Dockerfile. The build stage compiles the Rust binary in
# release mode (with the Linux-only `io_uring` cargo feature enabled by
# default — disable with `--build-arg FEATURES=""`). The runtime stage
# is a minimal `debian:bookworm-slim` image that carries only the
# binary, the example config, and a small entry-point script.
#
# Build:
#   docker build -t micro-expert-router .
#
# Run (with a host-side data dir mounted in):
#   docker run --rm -p 8080:8080 \
#     -v $PWD/data:/data \
#     -v $PWD/config.toml:/etc/mer/config.toml \
#     micro-expert-router serve --config /etc/mer/config.toml
#
# `docker compose up` is the simpler entry point — see docker-compose.yml.
# --------------------------------------------------------------------

FROM rust:1.83-bookworm AS build
ARG FEATURES="io_uring"
WORKDIR /src

# Cache cargo deps. We copy only Cargo.toml first so unrelated source
# changes don't blow the dependency layer.
COPY rust-engine/Cargo.toml rust-engine/Cargo.toml
RUN mkdir -p rust-engine/src \
    && echo "fn main() {}" > rust-engine/src/main.rs \
    && (cd rust-engine && cargo build --release ${FEATURES:+--features ${FEATURES}} || true)

# Bring in the real source.
COPY rust-engine/src rust-engine/src
RUN cd rust-engine \
    && cargo build --release ${FEATURES:+--features ${FEATURES}} \
    && cp target/release/micro-expert-router /usr/local/bin/micro-expert-router

# --------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# `ca-certificates` for any future HTTPS-fetched models;
# `curl` is used by the compose healthcheck (see `docker-compose.yml`).
# `libgcc-s1` is already present in slim but listed for completeness.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. The data dir is mounted in by the operator at runtime.
RUN useradd --system --create-home --uid 1000 mer
USER mer
WORKDIR /home/mer

COPY --from=build /usr/local/bin/micro-expert-router /usr/local/bin/micro-expert-router
COPY config.toml /etc/mer/config.toml

# Default to the HTTP server. Override the CMD to run benchmarks
# (`micro-expert-router run --data-dir /data --tokens 1000 …`).
EXPOSE 8080
ENV RUST_LOG=info
ENTRYPOINT ["/usr/local/bin/micro-expert-router"]
CMD ["serve", "--config", "/etc/mer/config.toml"]
