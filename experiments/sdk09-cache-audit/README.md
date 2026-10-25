# Read-only SDK 0.9 embedded-kernel audit

`audit_registry.py` compares the device images actually embedded in preserved
ELFs. It uses Python's standard library and opens ELF inputs read-only. It does
not invoke a compiler, subprocess, `objcopy`, SDK launch, or file-write API.
JSON goes only to stdout. Redirect it to a **new report file**, never an ELF.
Input SHA256 and metadata are checked again after parsing to detect concurrent
changes. Reading may update filesystem access time; contents and modification
time are not changed.

The parser follows installed
`furiosa-opt-std-0.9.0/src/backend/npu/registry.rs:10`: the
`furiosa_kernels` section contains bincode
`Vec<(function_path, Vec<u64> generic_key, image_bytes)>`. It supports the actual
ELF64 little-endian artifacts, checks lengths, rejects duplicate identities and
trailing bytes, and rejects extended ELF section numbering explicitly. It
compares by `(function_path, generic_key)`, not registry order. Reports include
whole-ELF, section, and per-image hashes. An image hash includes the entire
compiled image, not just its instructions.

## Reproduce the preserved audit

Run on the host; no Docker, Rust, or NPU is required. Set the checkout root for
your machine. The paths under it name immutable captured artifacts, not current
target outputs.

```bash
cd /home/jihwan38/moa-worktrees/round2/b16-f4-mse-lm-fp8/experiments/sdk09-cache-audit
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_audit_registry.py
python3 audit_registry.py known-suite \
  --round2-root /home/jihwan38/moa-worktrees/round2 \
  > /tmp/sdk09-registry-reproduced.json
```

The expected common/different counts are:

| Comparison | Common identities | Different images |
| --- | ---: | ---: |
| q021 explicit-control model versus activated server | 305 | 0 |
| physical32 versus physical64 physical-pool probe | 21 | 19 |
| physical32 versus physical64 history model | 301 | 50 |
| physical32 versus physical64 grouped model | 301 | 50 |

`known-suite` exits nonzero if those count expectations fail. Inspect the emitted
artifact hashes too: matching counts alone do not identify a build. The q021
model ELF must be `15ff8c81dc6db22a07069ec3790ad3b3f1d7cf79a5be13375d0cd18c08afa2c4`,
and the server must be
`7d89bb85e53fd9cd1cfcc44ebeaed4226aafe5f9cdb2063ac270ddcf4a272072`.
`reproduction-summary.json` records all eight input hashes from this rerun.

For other artifacts:

```bash
python3 audit_registry.py inspect /path/to/preserved.elf > /tmp/registry.json
python3 audit_registry.py compare /path/to/control.elf /path/to/candidate.elf \
  > /tmp/registry-pair.json
```

The ordinary `compare` command reports differences without deciding whether
they are expected. A zero exit status means parsing/comparison succeeded, not
kernel correctness or cache correctness.

## Fresh kernel build recipe (documentation only)

SDK 0.9 `cargo furiosa-opt --help` documents `FURIOSA_OPT_OUT_DIR`; its default is
`<workspace target>/furiosa-opt/kernel`. Use a **new, empty, uniquely named**
directory for every device-global-constant variant. Preserve the previous
directory and captured artifacts. Cargo package freshness and the SDK kernel
cache are separate: `--fresh-package` plus recorded source hashes alone did not
invalidate the observed external-constant dependency.

The repository's existing recorder can force the package build and capture a
separate executable. Substitute an owned worktree, unique run, and actual binary;
coordinate shared target ownership first. This audit did not execute this recipe.

```bash
python3 tools/record_run.py \
  --name UNIQUE_RUN --worktree OWN_WORKTREE \
  --purpose 'Isolated device-global-constant variant with fresh SDK kernel directory' \
  --fresh-package --capture-artifact target/release/ACTUAL_BIN -- \
  docker exec --workdir /round2/OWN_WORKTREE \
    -e CARGO_INCREMENTAL=0 -e CARGO_BUILD_JOBS=4 \
    -e CARGO_TARGET_DIR=/round2/OWN_TARGET \
    -e FURIOSA_OPT_OUT_DIR=/round2/runs/UNIQUE_RUN/kernel-out \
    moa2026-round2 cargo furiosa-opt build --release --bin ACTUAL_BIN
```

Before starting, verify the chosen `kernel-out` path does not exist. Record the
source/config hashes, complete command, actual device `Compiling` lines, resulting
kernel images and captured ELF hashes. Do not modify captured ELFs to inspect
them: this parser reads the section directly. Preserve both positive and failed
negative controls.

## Targeted negative-control requirement

For a device-global flag such as fixed-E4 `ENABLED`, independently build OFF and
ON with different fresh kernel directories. Use deterministic finite inputs
that exercise values changed by the codec. OFF must reproduce the unmodified
BF16 producer; ON must match an independent rounded-codec CPU oracle and must
actually change at least the chosen sensitive outputs. Exercise each relevant
producer/consumer route; a separate head probe cannot by itself prove that a
whole-model ELF includes the same rebuilt producer. Inspect that whole-model
ELF's affected entries before interpreting OFF/ON model results.

When a flag changes generated arithmetic, identical affected image bytes are a
stop condition until explained. Conversely, different images alone do not prove
correctness: metadata can differ without the required arithmetic changing.
Require the targeted negative and positive controls, then the intended model
gate. Root remains the sole Arena dispatcher.

## Scope of the earlier result

q021 changes a **host** LM selector between distinct, already-tested named
functions. Its model/server share all 305 common images byte-for-byte; the two
q020 imports are CPU weight-preparation code. That specific delta does not rely
on reevaluating a device-global constant inside an unchanged image.

Physical32 changed a named axis type/size; the preserved artifacts demonstrably
contain changed pool kernels, and resource/actual memory gates confirm the
smaller pool. Concrete device functions can still have empty runtime generic
keys, so do not say every shape change is encoded in those keys.

The installed compiler driver is prebuilt, and its cache implementation source
was unavailable. Runtime `(path,key)` selection is verified public code, **not**
the complete compiler fingerprint algorithm. The fixed-E4 evidence establishes
stale reuse for one external-constant edit; these comparisons neither prove nor
disprove all historical kernel-cache validity. Root preserved the bounded audit
report at commit `c8dbcce9`, `evidence/sdk09-cache-audit`.
