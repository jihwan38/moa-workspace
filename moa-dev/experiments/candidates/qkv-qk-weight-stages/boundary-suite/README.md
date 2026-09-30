> Current result: raw9 job137026 and full2 job137031 passed. See [result-summary.md](result-summary.md). The preparation/build records below retain their historical scope.

# E146 changed-boundary validation — prepared, not built or tested

This suite preserves the successful E144 raw9 wrapper, Rust host and F64 generator byte-for-byte. The original E53 full2 host, CPU reference, fixture, collector, tolerances and cache checks also remain unchanged. It contains no historical passing result copied as an E146 result.

`preparation.json` binds immutable test inputs and tooling. `verify_preparation.py --check` runs offline and accepts an unselected production variant. The final production selection is a separate exclusive-write `production-selection.json`; it must be created only after the parent chooses the variant and its actual graph gate passes. Failed v1/v2/v3 builds are not silently bound. This preparation was made while E146 exact compilation was managed separately; no build or Arena run was performed by these tools during preparation.

After parent selection, bind the exact label:

```bash
python3 moa-dev/experiments/candidates/qkv-qk-weight-stages/boundary-suite/select_production.py --build-label CHOSEN_PRODUCTION_LABEL
python3 moa-dev/experiments/candidates/qkv-qk-weight-stages/boundary-suite/verify_preparation.py --check --require-selected
```

The selector verifies the existing production graph gate, all46 compilation inputs, entire selected `projection.rs`, inherited `ops.rs`, actual whole exact-image file and build-record digest. There is no default production variant. Neither selection nor preparation compiles or submits a job. Once any boundary build/result exists, preserve the suite as frozen; a source/build repair needs a separate suite/attempt rather than overwriting records.

Raw9 invokes the unchanged production `project_qkv_native` on stored BF16 post-preRMS inputs and observes BF16 Q/K/V before head RMS. The9 original cases cover zero, K959/960/3839, row/group/cluster routing, tiny cancellation, scale-before-BF16 and a nonzero fourth term. F64 truth checks exact F32 representability then BF16 RNE. All73728 outputs must match exactly, with signed zero equivalent and no nonfinite values. All63 regenerated inputs must match original checksums and SHA256. Original truth bytes must also match, not merely a regenerated result count.

After a positive whole screen and explicit target/Arena coordination:

```bash
python3 moa-dev/experiments/candidates/qkv-qk-weight-stages/boundary-suite/raw_oracle.py build --build-label CHOSEN_PRODUCTION_LABEL --target ASSIGNED_TARGET
# Once explicitly authorized: raw_oracle.py submit, then collect, then --check.
```

Raw build copies the selected whole crate, preserves all original source bytes, appends only the diagnostic module declaration and adds the two original diagnostic files. Its exact diagnostic image is captured separately and matched to the host-embedded diagnostic image. The diagnostic image differs from the whole production image; source binding is the bridge. Actual uploaded tar bytes, raw log, complete input checks and all27 output comparisons are retained.

Full2 uses the original whole-production host; there is no device wrapper or kernel arithmetic change. Only `src/bin/probe_qkv.rs` is added to the selected crate. The resulting QKV image must equal the chosen whole exact image and occur exactly once in the host. The original acceptance remains atol0.04 / rtol0.01,16384 outputs,38 input hashes and8380416 untouched cache values.

```bash
python3 moa-dev/experiments/candidates/qkv-qk-weight-stages/boundary-suite/build.py --build moa-dev/experiments/candidates/qkv-qk-weight-stages/build-CHOSEN_PRODUCTION_LABEL-repro.json --label e146-boundary-v1 --target ASSIGNED_TARGET
# Root submits that package with name beef-e146-boundary-v1.
python3 moa-dev/experiments/candidates/qkv-qk-weight-stages/boundary-suite/configure_full.py --job JOB
python3 moa-dev/experiments/candidates/qkv-boundary-probe-e53/collect_qkv.py --config moa-dev/experiments/candidates/qkv-qk-weight-stages/boundary-suite/collector-config.json
python3 moa-dev/experiments/candidates/qkv-qk-weight-stages/boundary-suite/check_full.py
```

The copied full2 fixture is138912B, SHA256 `fad33f6da059c34bbdb69b9817a3bbe1ef6d2dadfad59f6b134ef2d4cca848ed`. The full2 cases are outside official PRNG support; raw9 is a finite set of mathematical checks. Neither establishes generic BF16 correctness. Standard performance and extra6 remain separate.

## Build reuse and required new work

- Reuse original raw9 source/expected-truth/input hashes and full2 host/reference/fixture. No new CPU-reference design or modified tolerance is needed.
- E144 baseline raw/full images and passing observations already exist. No new baseline compile or baseline-only boundary Arena job is needed; these are provenance/reference evidence, not inherited E146 correctness.
- The selected E146 whole exact image/crate already exists. Full2 only needs its original host built against that crate, with emitted-image identity checked. Do not separately recompile the original whole entry point just for this suite.
- Raw9 needs one new exact diagnostic compile and host build from the selected E146 source. An E144 diagnostic image cannot validate changed E146 stages. The warm target may reuse Cargo work, but the recorded image/source checks remain mandatory.
- Candidate standard hosts cannot substitute for raw/full hosts because they do not execute these unchanged boundary inputs or expose raw Q/K/V. Fixtures/reference bytes can be reused; a changed executable must be newly bound and checked.
