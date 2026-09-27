# E136 full-production QKV cancellation checks

`build.py` binds the frozen `qkv-boundary-probe-e53/probe_qkv.rs` host to the
selected exact E136 build. Its 46 original compilation inputs must be unchanged;
only `src/bin/probe_qkv.rs` is added. The emitted production QKV image must match
the exact candidate image and occur once in the uploaded host. The host invokes
`ops::sliding_project_qkv` directly, including pre-RMS, projection, scale,
head RMS, RoPE and cache stores. There is no device-side diagnostic wrapper.

The frozen original-reference fixture has two cancellation cases:
`tiny_lost_after_pre_rms` and `larger_tiny_preserved_control`. Their expected V
values are respectively 0.85546875 and 1.0. Both are outside the current public
QKV input generator's support. They are precision regression tests, not observed
official random seeds, a performance measurement or a proof for every input.
Every regenerated input is checked against the original CPU byte stream (38
SHA256 comparisons), and Q/K/V use the unchanged atol=0.04, rtol=0.01.

The NPU host checks 16,384 output values and 8,380,416 non-target cache values
after execution. Constant cache sentinels and identical weight rows limit what
these cases establish about row routing; standard random/extra-input validation
must cover the changed row ownership separately.

`build-e136-boundary-v1.json` records the production-image binding, 47 input
hashes, exact host build, fixture provenance and host-only checks. Source,
images, full dumped inputs and build logs remain in the shared ignored
`moa-dev/experiments/artifacts/qkv-native-weight-resident/e136-boundary-v1/`.
Arena results, when present, are separate records; a successful host build is
not an NPU precision result.
