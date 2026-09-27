# Same-input full FFN precision regression

Arena job **109965** ran the unchanged public `ops::decoder_feedforward` images from E128, E133 v2, and E134 weighted-v3 against one identical, correlated input. The expected output comes from the original official PyTorch `gemma4.MLP`, `GloballyScaled`, and RMS normalization implementations preserved under `upstream/`. No numerical tolerance was widened.

| Image | Exact BF16 outputs | Outputs outside original tolerance | Maximum absolute error |
| --- | ---: | ---: | ---: |
| E128 control | 3,840 / 3,840 | 0 | 0 |
| E133 v2 | 1,920 / 3,840 | 1,920 | 0.029296875 |
| E134 weighted-v3 | 3,840 / 3,840 | 0 | 0 |

The expected output is 0.404296875 throughout. E133 produces 0.375 in the first half: its same-scale E5 residual terms lose the small activations that the down projection selects. E134 restores those terms with coefficients `s`, `s/16`, and `s/256` before the F32 term reduction. The standalone down oracle separately checks the same mechanism before the FFN epilogue.

This is a constructed regression within the public generator's component value ranges. It is not claimed to be a generated public PRNG seed, an exhaustive BF16 guarantee, an end-to-end model test, or an official competition evaluation. This diagnostic's timing is excluded from performance evidence. The host intentionally completes all three stages even when E133 fails; `verify_results.py` requires E128 and E134 to pass and explicitly requires the E133 failure to reproduce.

`verify_results.py --check` rechecks the actual uploaded archive, production-image identity, source/reference/build hashes, raw Arena stage results, all 13 input tensor hashes, and the three Rust preparation fingerprints. `verify_reference.py` binds the official Torch reference and expected BF16 bytes. `verify_input_codegen.py` independently reconstructs the packed inputs and compares their hashes with the Torch reference and actual Rust host logs.

The first builder, `run.py`, compiled the E128 host successfully but looked for its image at a flat path. `build-e128.json` preserves that capture failure. `run_v2.py` fixes discovery of the nested kernel image; `build-e128-recovered.json` captures the already built image without rebuilding or changing source. The initial submission attempt returned nonzero without a captured explanation; an Arena job-list check found no accepted job with this name. The explicitly recorded retry accepted job 109965. No cause is inferred from that unexplained failure.

The source, failed attempts, build records, reference, raw results, and verification records are retained. Reproduce the evidence check from the workspace with:

```sh
python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/full-ffn-stress/verify_results.py --check
```
