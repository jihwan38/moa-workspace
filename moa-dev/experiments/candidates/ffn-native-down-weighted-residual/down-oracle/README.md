# E134 weighted down oracle

All nine cases are required exact BF16 checks (signed zero equivalent). The
former E133 adversarial case4 has no exemption: the weighted residual repair
must recover its tiny contribution. Cases0–7 reuse the exact prior original
BF16 inputs/weights/scales/reference; only the candidate source changes.
Case8 adds the raw-down boundary from the component-range CPU FFN construction:
L-half anchors1024/0.5, all other values2^-22, down blockscale15; first1920 output
rows select cluster0 tiny values, last1920 select cluster1 tiny values, weight6,
anchor weights0. Both reference output halves round toBF16 0.1650390625.

The wrapper copies the actual frozen production MLP byte-for-byte and appends
only a diagnostic entry. Input BF16 L15360 is placed into the original M03
LocalRows layout; `project_down_tile` performs the real weighted quantizer,
term copies/broadcast, FP4 native contraction, per-term restore and F32 partial
reduction; `finish_down` completes cluster sum and the original BF16 boundary.
No fixture-specific computation enters the device source.

The host reference reads the original BF16 vector and independent FP4 value
specification. It uses F64 arithmetic with exact-F32 assertions after every
block product/accumulation, blockscale product and column/cluster add on these
chosen dyadic inputs. It never compares against its own E5 reconstruction.
The original standard compare body and0.01 absolute/0.01 relative constants are
preserved and checked; all nine tests also require exact BF16 equality, so
absolute tolerance cannot conceal a zero or tiny mapping error. Output buffers
start with NaN sentinels and nonfinite/missing output is a failure.

The probe covers34560 outputs. Case8 is not a full-FFN NPU replay: later global
scale, RMS/residual/layer scale, and pre-GeGLU reachability are outside this
entry. It neither proves a public PRNG seed nor general BF16/model correctness.
Complete-kernel standard/extra inputs and performance remain separate evidence.

The builder binds the candidate exact build and graphgate before compiling,
verifies original source/container hashes, exact-vs-host image equality and CPU
oracle-only outputs, and freezes all inputs. The runner preserves actual upload
archive and raw NPU logs. A launch timeout is separate from arithmetic mismatch.
Target `/tmp/moa-m03-validation-probe/target` belongs to this oracle while used;
whole candidate compilation uses a separate target owned by literature.

```sh
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/down-oracle/probe.py build
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/down-oracle/probe.py submit
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/down-oracle/probe.py collect
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/down-oracle/probe.py --check
```

Root authorized one bounded oracle allocation after the new graphgate. This is
a diagnostic-only image, never a performance sample or official submission.

The executable builder selects `recipe-weighted-v3.json`, bound to the final
whole-compile-passing source. The original `recipe.json` remains the immutable
v1 preparation input used by `feasibility/record.json`: its standalone exact
compile failed with the same SFR868 error as the full v1 kernel, and no host or
NPU ran for that attempt. Weighted-v3 changes the F32 sum and coefficient
producer contexts to Main without changing the weighted formula or layout.
The CPU oracle also checks that the first eight inputs/weights/scales/reference
checksums equal the original frozen E133 oracle, so its adverse case is retained.
