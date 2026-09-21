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

Follow [the competition README](../furiosa-opt-gemma4-12B/README.md) for the
allowed changes and [OPTIMIZATION.md](../furiosa-opt-gemma4-12B/OPTIMIZATION.md)
for the optimization workflow.
