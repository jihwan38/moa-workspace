> Current result: raw9 job137109 and full2 job137116 passed. See [result-summary.md](result-summary.md). Preparation/build scope below is historical; no build record was rewritten.

# E147 boundary validation preparation

Status: prepared, no selected production build, no boundary compile or NPU. Source/control is E144; no E146 production kernel or passing result is reused. The original raw9 wrapper, Rust host and F64 generator are copied byte-for-byte from E144. Source-neutral selection/build/collection tools reuse the validated E146 machinery with a new namespace.

`verify_preparation.py --check` checks original raw source identity, full2 fixture/reference/tolerances and tooling hashes offline. Only after parent selection and actual graph gate PASS may `select_production.py --build-label LABEL` create immutable `production-selection.json`. It checks all46 production inputs, actual whole exact image, complete projection/ops sources and the graph gate. Compilation and Arena still require explicit root coordination; m03 is reserved only when root authorizes these boundary builds.

Raw9 tests unchanged production `project_qkv_native` from post-preRMS BF16 input through the original F32 common/channel scale and first BF16 outputs before head RMS. All73728 outputs must match BF16 exactly, signed zero equivalent and nonfinite0; all63 input SHA256 hashes must match the original independent F64 truth. Four-term-active, cancellation, scale-before-BF16, K and row/group/cluster cases are unchanged. The raw diagnostic image differs from the whole image; the full46 production sources remain bound, with only the documented lib module append and two original diagnostic files added.

Full2 uses the original E53 host, fixture, F64/CPU reference and collector, with atol0.04 / rtol0.01. Its host must embed the exact selected whole QKV image once. Required checks are16384 outputs,38 input hashes and8380416 untouched cache values. The2 cancellation cases are outside public PRNG support. Neither raw9 nor full2 is a general all-BF16 proof or a performance measurement.

After each explicit authorization, from the repository root:

```bash
python3 moa-dev/experiments/candidates/qkv-scalar-sub-vrf/boundary-suite/verify_preparation.py --check
python3 moa-dev/experiments/candidates/qkv-scalar-sub-vrf/boundary-suite/select_production.py --build-label SELECTED_LABEL
python3 moa-dev/experiments/candidates/qkv-scalar-sub-vrf/boundary-suite/raw_oracle.py build --build-label SELECTED_LABEL --target ASSIGNED_TARGET
# raw_oracle.py submit / collect / --check only after NPU authorization.
python3 moa-dev/experiments/candidates/qkv-scalar-sub-vrf/boundary-suite/build.py --build moa-dev/experiments/candidates/qkv-scalar-sub-vrf/build-SELECTED_LABEL-repro.json --label e147-boundary-v1 --target ASSIGNED_TARGET
# Submit the full2 package once with name beef-e147-boundary-v1, then:
python3 moa-dev/experiments/candidates/qkv-scalar-sub-vrf/boundary-suite/configure_full.py --job JOB
python3 moa-dev/experiments/candidates/qkv-boundary-probe-e53/collect_qkv.py --config moa-dev/experiments/candidates/qkv-scalar-sub-vrf/boundary-suite/collector-config.json
python3 moa-dev/experiments/candidates/qkv-scalar-sub-vrf/boundary-suite/check_full.py
```

The original full2 fixture is138912B, SHA256 `fad33f6da059c34bbdb69b9817a3bbe1ef6d2dadfad59f6b134ef2d4cca848ed`. It is copied locally so builds need no ignored legacy fixture path. Previous baseline boundary images/results need no recompilation, but cannot validate E147. A new raw diagnostic exact+host build and a new full2 host bound to the already-built production image are required. Standard/extra6/performance records remain separate. Preserve all build/result records; a changed source attempt needs a new suite rather than an overwrite.
