# Independent Rust-host/Torch input cross-check

`verify_input_codegen.py` independently reconstructs the frozen Rust host's
packed-byte expressions and constants. All13 tensor SHA256 values match those
recorded by the actual official dense PyTorch reference execution. The FP4
packing is low nibble first; local scales are encoded2/4/15, and all three
global tensors hold the reciprocal2^-11. E128/E133/E134 actual `--prepare-only`
logs each match the independently computed FNV1a line:

```
FULL_STRESS_INPUT up=7cc5b3df0410900d gate=3845ea07dc062539 down=96568eed1d9a2325 local=a60bacb398262325 expected=5b64d68f7e4b5d25
```

`local` fingerprints decoded scale F32 bytes, matching the host's
`f8e4m3::to_f32().to_bits()` contract. Other tensor byte equality is checked
against the reference and frozen source; the host does not print runtime
hashes for its remaining constant tensors. This limit is explicit in
`input-codegen-verification.json`.

The host checks every one of3840 typed BF16 output elements, rejects nonfinite
values and keeps `abs(actual-expected) <= 0.01 + 0.01*abs(expected)`.
Exact equality is additionally reported. A process exit0 is not itself the
correctness gate: the collector separately enforces E128 and repaired E134
acceptance while preserving the expected E133 diagnostic failure.

```sh
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/full-ffn-stress/verify_input_codegen.py --check
```

The checker compiles only the tiny `fnv64.c` CPU helper in an automatically
removed temporary directory. It uses no Cargo target, NPU compiler or device.
It verifies all three frozen prepare logs even without optional arguments.
