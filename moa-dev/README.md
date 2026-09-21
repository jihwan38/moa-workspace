# MOA 2026 development environment

The competition crate is `~/moa-workspace/furiosa-opt-gemma4-12B`. It and
`moa-dev` belong to the single Git repository at `~/moa-workspace`.
The `moa2026-dev` container mounts the competition directory at `/workspace`,
so edits on the host and build outputs in the container use the same files.
The container runs as UID/GID 1000 to keep generated files owned by the host user.

Enter the development shell from a host terminal:

```bash
~/moa-workspace/moa-dev/moa
```

Or run one command directly:

```bash
~/moa-workspace/moa-dev/moa cargo furiosa-opt build --release --bin test_kernels
```

The wrapper starts the existing container, or creates it from the
`moa2026-dev:0.8.1` image if needed. Rebuild the image with:

```bash
docker build -t moa2026-dev:0.8.1 ~/moa-workspace/moa-dev
```

## Recorded baseline environment

The installed image used for successful RNGD baseline jobs is
`sha256:518d7b7ddaca5a97344d8bbd60e04f5e87ca7c50d883bfb5930a64a0c6ff382f`.
The running container uses that same image ID. Its observed versions are:

| Component | Installed version |
|---|---|
| Ubuntu / glibc | 22.04 / 2.35 |
| Rust toolchain | `nightly-2026-05-01` (`rustc 1.97.0-nightly`, commit `f53b654a8`) |
| cargo-binstall | `1.23.0` |
| cargo-furiosa-opt | `0.8.1` |
| furiosa-arena-cli | `0.8.0` |
| moa-submitter-cli | `1.2.0` |
| furiosa-schedule-viewer | `0.3.0` |
| Python | `3.10.12` |
| torch / numpy / safetensors | `2.14.0+cpu` / `2.2.6` / `0.8.0` |

[requirements-cpu.txt](requirements-cpu.txt) pins all 13 installed Python
distributions, including transitive dependencies and packaging tools. The
Dockerfile now pins the installed Rust CLI versions too. These changes do not
update the existing image or running container. On September 21, 2026, the
pinned recipe was rebuilt without cache as `moa2026-repro-audit:0.8.1`.
Its clean `--locked` build of the saved E18 source produced a **byte-identical
official test binary**. See the [fresh-build audit](../docs/experiments/repro-audit-e18.md)
for hashes, installed versions, and limits. The fresh container was not used
for NPU execution; the existing development image and container remain unchanged.

[environment-baseline.json](../docs/experiments/environment-baseline.json)
records the image identity, complete observed apt/Python package versions,
installed Rust executable hashes, Python package metadata/RECORD hashes, and
the recipe and Cargo lockfile hashes. Python metadata/RECORD hashes describe
the installed distributions; they are not wheel download hashes or a
`pip --require-hashes` lock.

The competition crate ignores `Cargo.lock`. Its tested dependency resolution is
preserved separately in [locks/gemma4-Cargo.lock](locks/gemma4-Cargo.lock). On a
fresh checkout, restore it on the host before building, without replacing an
existing lockfile:

```bash
test -e ~/moa-workspace/furiosa-opt-gemma4-12B/Cargo.lock || \
    cp ~/moa-workspace/moa-dev/locks/gemma4-Cargo.lock \
       ~/moa-workspace/furiosa-opt-gemma4-12B/Cargo.lock
```

Record any intentional dependency update with a new lockfile snapshot and a
fresh hardware baseline. Do not silently compare it against measurements made
with a different compiler or dependency resolution. The pinned recipe still
uses the floating `ubuntu:22.04` tag, Ubuntu apt repositories, the rustup
installer, and external package servers. It does not promise a byte-identical
image rebuild; the recorded image ID identifies the environment already tested.

## Build and measurement

The following commands run **inside the development shell**, at `/workspace`.
Generate the expected outputs and build the original kernel tests:

```bash
python scripts/generate_references.py
cargo furiosa-opt build --release --bin test_kernels
```

Authenticate with the GitHub account used on the registration form, then run
the already-built test on Arena:

```bash
furiosa-arena login
RNGD_SUBMIT_TIMEOUT=70 ./scripts/rngd_test.sh --no-build
```

Arena rejected the script's default 120-second job timeout on September 21,
2026 because the server maximum was 70 seconds. `RNGD_SUBMIT_TIMEOUT=70`
sets the remote job's execution limit; the script still waits up to 1,800
seconds for the queued job to finish. This setting uses the existing script
option without changing the competition repository.

After changing kernel code, omit `--no-build` so the script rebuilds it:

```bash
RNGD_SUBMIT_TIMEOUT=70 ./scripts/rngd_test.sh
```

To inspect a static schedule locally:

```bash
mkdir -p target/schedules
cargo furiosa-opt compile ops::sliding_attention_output --exact \
    --dump-schedule target/schedules/sliding_attention_output.json
furiosa-schedule-viewer --host 0.0.0.0
```

The wrapper publishes the viewer only on the host's `127.0.0.1:9254`. Open
`http://127.0.0.1:9254` on this machine and load the JSON from the repository's
`target/schedules` directory. Static schedules help diagnose a bottleneck;
Arena tests provide hardware correctness and cycle measurements.

Official grading is a separate command, run when ready to submit:

```bash
moa-submitter login
moa-submitter submit
moa-submitter status
```

The [E55 local CLI preflight](experiments/candidates/official-submit-preflight-e55/README.md)
ran the installed submitter against a loopback receiver and verified that all
23 uploaded source files match the measured E55 sources. It also checked source
discovery from the crate, a nested directory, and an explicit `--source` path.
This verifies packaging only; official authentication, server contract checks,
and grading still require the competition service.

Follow [the competition README](../furiosa-opt-gemma4-12B/README.md) for the
allowed changes and [OPTIMIZATION.md](../furiosa-opt-gemma4-12B/OPTIMIZATION.md)
for the optimization workflow.
