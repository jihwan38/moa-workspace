# E151 changed-boundary results

**Both original boundary validations pass.** These are separate correctness diagnostics; their timing is excluded from performance and they do not themselves decide adoption.

- Raw down job141395: all nine original E134 cases,34560 exact BF16 outputs, signed zero equivalent. Full frozen E151 MLP prefix, original wrapper/host, all original nine CPU input/projection checksum records, exact diagnostic image and actual uploaded archive are verified. The separate raw image is not the public FFN performance image.
- Whole FFN job141409: the original correlated precision-regression input,3840/3840 exact outputs, zero nonfinite values and zero maximum absolute error. Expected output remains0.404296875. The host includes the exact production FFN image; all46 original production files, original official Torch reference, unchanged tolerance0.01+0.01*abs(expected), original13input checks and actual uploaded archive are verified. This constructed input is not claimed to be a public PRNG seed.

`boundary-results.json` binds the frozen source, raw/full result records and verifiers. Independent raw graph review confirms the new scale ingress and five Sub Switch-to-VRF producers against the actual whole graph, as well as the two native half readers and full partial offsets0/14400B, including the unchanged12-row consumer spanning rows24..35 and shared register lifetimes. No generic BF16, end-to-end model or rank guarantee follows from these selected cases.

The original up/gate245760-value probe was not rerun and is not counted as this candidate's new boundary result. No inputs, references or tolerances were modified. m03 target is released; Arena slot2 was handed to root for the separately selected E152 first whole screen. No further E151 boundary jobs are active.

Recheck the actual results:

```sh
python3 moa-dev/experiments/candidates/ffn-down-scale-ingress-switch/down-oracle/probe.py --check
python3 moa-dev/experiments/candidates/ffn-down-scale-ingress-switch/full-ffn-stress/verify_results.py --check
```
