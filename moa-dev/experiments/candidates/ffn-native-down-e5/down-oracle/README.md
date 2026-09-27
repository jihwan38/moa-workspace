# E133 actual-down boundary oracle

This is a separate diagnostic image, excluded from performance and official
submission. The frozen v2 production MLP is included byte-for-byte; only a
wrapper is appended. It loads synthetic BF16 L15360 into the existing M03
cluster/local-row layout and calls the actual `project_down_tile` then
`finish_down`. Quantization, term copies, broadcast, native dot, both scales,
F32 column/cluster reductions and the final BF16 cast all execute on NPU.

The host computes reference values from original BF16 inputs, independent FP4
code values and original FP8 blockscales. It never substitutes E5 reconstructed
inputs. The F64 computation asserts exact F32 representability after every
block dot/add, blockscale product, column add and cluster add on these dyadic
cases. This avoids making an F64-vs-F32 rounding difference into a layout claim.
The standard host `compare` body is copied verbatim and its FFN constants
`atol=0.01, rtol=0.01` are checked against the frozen production host. Applying
those constants to an internal down output is diagnostic, not the unchanged
public whole-FFN contract, which includes later scale/residual/RMS operations.

Cases0/1/2/3/5/6/7 check exact BF16 outputs (signed zero equivalent): dense codes
and different cluster scales; isolated K16/64/480/1920/7680/end positions;
block cancellation; zero; representable tiny inputs; cross-cluster cancellation;
and distinct cluster scales with the large amax terms assigned zero weight.
All FP4 codes including signed zeros are covered by dense/zero cases. Shape is
full3840×15360; sentinel outputs expose missing stores. Seven cases cover26880
exact mapping/algebra outputs.

Case4 quantifies E5 underflow: cluster amax128 versus0.5, every other value2^-24,
large amax entries with zero weight, tiny entries with FP4=6 and blockscale256.
The original BF16 output exceeds1, so a raw0.01 absolute tolerance cannot hide
lost tiny terms. This case is adversarial and synthetic. Its failure is recorded
separately and requires a reachable full-FFN test under the original acceptance
before deciding adoption. These BF16 vectors are not claimed to be proven
post-GeGLU values for the official input generator. A finite-output failure in
this diagnostic cannot be silently waived as quantization. No public tolerance
or fixture is modified.

`probe.py build` creates a dedicated crate, verifies source/container hashes,
compiles the exact kernel and host, checks exact-vs-embedded image equality and
runs CPU oracle-only validation. `submit`, `collect`, and `--check` preserve the
uploaded archive and raw log. A failed or incomplete launch is separate from
observed arithmetic mismatch. The warm target is
`/tmp/moa-m03-validation-probe/target`; its previous owner explicitly released it.

```sh
python3 moa-dev/experiments/candidates/ffn-native-down-e5/down-oracle/probe.py build
python3 moa-dev/experiments/candidates/ffn-native-down-e5/down-oracle/probe.py submit
python3 moa-dev/experiments/candidates/ffn-native-down-e5/down-oracle/probe.py collect
python3 moa-dev/experiments/candidates/ffn-native-down-e5/down-oracle/probe.py --check
```

Preparation was authorized before the whole screen. Compilation/NPU was
subsequently authorized by root after whole job109740 measured−3.82638% with45
comparisons PASS. Generated `build.json`/`results.json` establish actual execution
status; source preparation alone is not a correctness or performance result.
