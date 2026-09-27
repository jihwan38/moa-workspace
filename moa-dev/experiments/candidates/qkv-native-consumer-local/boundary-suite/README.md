# E138 actual whole-QKV boundary validation

This suite reuses the immutable E53 two-case host and CPU-reference fixture;
only its production build binding changes. `build.py` adds one host source to
the selected E138 exact-build crate, checks all46 original source inputs,
requires the actual whole-QKV image to equal the exact compiled image and occur
once in the host, and verifies all38 generated input SHA256 values against the
original CPU source. It invokes `ops::sliding_project_qkv`, including pre-RMS,
all Q/K/V projections, channel scaling, head RMS/RoPE and cache writes.

The two cases are finite cancellation regressions outside the original PRNG
support, not all-input proofs. Identical weight rows and constant cache
sentinels limit routing coverage: standard random and emitted coordinate
checks must cover the changed H960/K4/row ownership separately. Original
atol0.04/rtol0.01 and reference/gemma4.py semantics remain unchanged. These
checks are correctness only; adoption performance uses complete standard hosts
with contemporary E137 controls.

Completed against frozen native-v1: job111211 passed16384 output checks under
original tolerances and8380416 untouched-cache checks. The exact whole image is
188672B, SHA256a4e30799e7c9c49f1fe79c74513da2abe5760795d9fe1f31bc2bb16fb3ee79b7.
Actual uploaded package and all38 generated input SHA256 values were verified.
This does not change the whole-kernel performance regression in job111168.

`raw_oracle.py` separately compiles the unchanged production
`project_qkv_native` helper, with a wrapper supplying its actual post-preRMS
BF16H3840 boundary and collecting Q/K/V before head RMS. All original source
files remain byte-identical, except appending the wrapper module to lib.rs;
the raw image is a diagnostic, not the whole QKV image. Its nine independent
F64-reference cases cover signed zero, K959/960/3839 and all K4 boundaries,
Q32/KV16 group and cluster boundaries, correlated tiny cancellation, scale
before the first BF16 cast, and a constructed fourth-residual-active input.
Reference decoding uses stored BF16 and original E4 weights and F32-exact
mathematical sums, not the candidate's residual quantizer. The fourth-term case
has tiny normalized terms[0,0,128,32]; a three-term implementation is insufficient.

Job111216 passed all73728 raw BF16 outputs exactly (signed zeros equivalent).
The host-generated63 inputs matched independently generated full SHA256 values;
actual uploaded archive and embedded diagnostic image were verified. The
post-preRMS cases are not claimed to be reachable public-generator seeds.
This finite set and standard whole inputs do not prove arbitrary-input accuracy.

Recheck raw source/package/reference/results without a new Arena job:

```sh
python3 moa-dev/experiments/candidates/qkv-native-consumer-local/boundary-suite/raw_oracle.py --check
```

The full2 result, immutable original collector, config, raw log and uploaded
archive are retained. `check_full.py` verifies their recorded binding offline.
There was no extra6 adoption expansion for this measured slower prototype.
