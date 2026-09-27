# E133 numerical scope: measured helper loss and a full-FFN CPU construction

E133's seven down mapping cases pass exactly, but its unchanged three-term, common-scale E5 representation is not an exact encoding of arbitrary BF16 activations. The measured helper case4 loses half the raw output. The CPU construction below uses values within the public generator's component ranges and predicts a final full-FFN error above the unchanged tolerance after post-RMS and residual addition. This construction has **not** been run as a complete PyTorch fixture or on NPU, and no public PRNG seed producing it has been found. It is a concrete numerical risk, not an observed public grading failure or a new adoption decision.

`numerical_scope_analysis.py` reproduces the scalar calculation with standard-library F32/BF16 rounding, E5M2 RNE and both GeGLU forms; `numerical-scope-analysis.json` binds its sources by digest. Run from the workspace root:

```sh
python3 moa-dev/experiments/candidates/ffn-native-down-e5/numerical_scope_analysis.py --check
```

The frozen production source is [down-e5-v2/mlp.rs](down-e5-v2/mlp.rs), SHA256 `c3ae74b55b306742b69b438bf2ea603409a0644c55f9004d7fe01665c656a5d3`. Contract files are bound to source baseline `cc82b9ccff22795165faffd3ba9a68cc557da001`, so a future live-source update does not silently change this analysis.

## What the public tests establish

The public [README](../../../../furiosa-opt-gemma4-12B/README.md#L34) identifies `src/bin/test_kernels.rs` as the Stage1 correctness/performance source of truth, and requires every independently seeded configuration to pass. The current official text was also reviewed by root at [the official repository](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B/blob/main/README.md). This note neither changes that acceptance rule nor claims an all-BF16 bitwise requirement.

In `test_kernels.rs`, `f8_banded` at lines105–120 chooses exponent bits8–10 and mantissa0–7 for unsigned FFN local scales: their values are **2 through15**, not256. `bf16_uniform_range` at85–90 produces residual/pre/post RMS weights from UNIT=[0,1); values near1 round to BF16 1. FP4 nibbles at123–130 permit all sixteen codes. Constants148–155 and FFN input assembly572–600 use raw global scales [2048,16384), their F32 reciprocals and layer scalar0.375. The FFN test keeps `atol=rtol=0.01` at462–463; comparison at626–651 checks every element and rejects nonfinite outputs.

Public reference semantics are explicit: [generate_references.py:358–413](../../../../furiosa-opt-gemma4-12B/scripts/generate_references.py#L358) folds FP4 and local scale into BF16 weights, computes BF16 linear outputs, applies global reciprocals in F32, then commits BF16. [gemma4.py:75–89](../../../../furiosa-opt-gemma4-12B/scripts/reference/gemma4.py#L75) defines F32 RMS with epsilon1e-6 and BF16 output; its MLP at189–197 uses BF16 `gelu(..., approximate="tanh") * up`. [ops.rs:228–247](../../../../furiosa-opt-gemma4-12B/src/ops.rs#L228) connects pre-RMS, FFN, post-RMS, BF16 residual addition and layer scaling.

Passing the three standard and six extra seeded inputs provides actual evidence for those nine inputs. It does not prove the correlated construction below occurs for any seed, and neither the scalar-range argument nor those nine passes proves end-to-end/model-wide safety.

## Actual helper evidence and mechanism

The immutable [down-oracle/results.json](down-oracle/results.json) records Arena109784, correct upload, seven mapping cases /26,880 exact BF16 outputs. Adversarial case4 has all3,840 outputs0.703125 against1.40625: 50% raw-down loss. Its blockscale256 is outside the standard generator's local-scale band, and its output is identical in every H row. A global multiplicative error of that form can mostly disappear in post-RMS; it alone is insufficient to establish public full-FFN failure.

Production `prepare_down_fp8_split` at [mlp.rs:282–383](down-e5-v2/mlp.rs#L282) chooses a separate scale for each7680-element L half:

```
s = 2^(floor(log2(max(amax, 2^-112))) - 14)
u = x/s
hi = E5(u); lo = E5(u-hi); third = E5(u-(hi+lo))
```

All three terms use the same scale. E5M2's smallest positive subnormal is2^-16. A normalized residual below2^-17 rounds to zero; evaluating another term at the same scale cannot recover it. `project_down_tile` at413–425 contracts `FfnTerm` away before applying the common restore scale. This is a representation limitation, not evidence of wrong row mapping.

## A component-wise in-range full-FFN construction

Set residual, pre-RMS weight and post-RMS weight to BF16 1. Set all three raw global scales to2048 and layer scalar0.375. Pre-RMS produces BF16 1 despite epsilon1e-6. Set up/gate rows using the following **FP4 values**, followed by zero weights to3840 elements; scale is uniform over each row's 16-element blocks.

| L row | Up weight recipe | Gate weight recipe | Local scale | U and G after global/BF16 | BF16 GeGLU |
|---|---|---|---|---|---|
| All except0,7680 | one value0.5 | one value1 |2|2^-11,2^-10|2^-22|
|0|2730 values6, one value4|same|4|32,32|1024|
|7680|five values6, one value2|682 values6, one value4|4|1/16,8|0.5|

The raw dots are respectively(1,2), (65536,65536), (128,16384), all exact in F32 and BF16 at the required output boundary. The public tanh GeGLU with its BF16 intermediate and the production erf GeGLU CPU model round to the same listed values. This statement is specific to these rows; it is not a general equivalence claim for the two approximations.

Cluster0 thus has amax1024 and tiny value2^-22: scale2^-4, normalized tiny2^-18, which vanishes **below** the RNE tie. Cluster1 has amax0.5: scale2^-15, normalized tiny2^-7, which is exactly E5-representable.

Make down blockscales15 for every block. For output rows h<1920, set FP4 value6 at L indices1..7679 and zero elsewhere. For h>=1920, set value6 at indices7681..15359 and zero elsewhere. Both anchor columns have zero down weights. This maps the two activation halves to different output rows, so the loss changes output direction instead of only its global magnitude. All scales2/4/15 and all weight/input values are component-wise in the generator's range; the deliberately correlated pattern is **not** a claimed realization of its PRNG.

| Boundary | BF16-reference CPU model | E133 CPU model |
|---|---:|---:|
| Raw down, either output half |7679×2^-22×6×15 =0.16477346420288086|first half0; second unchanged|
| Raw-down BF16 |0.1650390625|0 /0.1650390625|
| Global reciprocal and BF16 |0.000080585479736328125|0 /same|
| Post-RMS BF16 |0.08056640625 in both halves|0 /0.08056640625|
| Residual and layer output BF16 |0.404296875 in both halves|0.375 /0.404296875|

The first1920 outputs have predicted difference **0.029296875**, exceeding unchanged tolerance **0.01404296875**. This is a source-grounded scalar estimate with explicit BF16/F32 boundaries, not hardware validation. Hardware reduction order/unary approximations and the actual public PyTorch fixture have not been replayed for this construction. It is sufficient to prevent a claim that the common-scale representation generally preserves full-FFN behavior; it does not establish a failure of the fixed graded inputs.

## A coupled repair remains open

A concrete follow-up is to preserve `FfnTerm` in native F32 output, restore each term with its own scale, and only then sum terms. A smaller third-residual scale could retain values now lost to underflow. Its multiplier must be derived from the actual normalized BF16/F32 residual bound and E5 overflow limit; 256 or another constant must not be chosen by guess. Changing only the quantizer would be wrong because current `contract_time` at420 already sums equal-scale terms before restoration.

Such a repair adds a term-dependent F32 restore/reduction and may enlarge partial storage, packets or engine work. Public API/lowering support, full-boundary correctness and whole-FFN performance would require new evidence. This note does not close the native-down family, authorize a new kernel or replace the existing numerical acceptance gate.
