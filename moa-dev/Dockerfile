FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git libclang-dev \
    gcc-aarch64-linux-gnu pkg-config libssl-dev python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

ARG MOA_UID=1000
ARG MOA_GID=1000
RUN groupadd --gid "$MOA_GID" moa \
    && useradd --uid "$MOA_UID" --gid "$MOA_GID" --create-home --shell /bin/bash moa
USER moa
WORKDIR /home/moa
ENV PATH="/home/moa/.cargo/bin:/home/moa/.venv/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain nightly-2026-05-01 \
    && rustup component add rustfmt clippy
RUN cargo install cargo-binstall --locked
RUN cargo binstall --no-confirm cargo-furiosa-opt@0.8.1
RUN cargo binstall --no-confirm furiosa-arena-cli moa-submitter-cli furiosa-schedule-viewer

RUN python3 -m venv /home/moa/.venv \
    && python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir numpy safetensors

# gcc-aarch64-linux-gnu only recommends its libc headers; install them explicitly
# because the base package installation uses --no-install-recommends.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libc6-dev-arm64-cross \
    && rm -rf /var/lib/apt/lists/*
USER moa

WORKDIR /workspace
CMD ["sleep", "infinity"]
