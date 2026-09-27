# E137 native-v3 validation follow-up

The consumer-local quantization repair passes the same finite O checks as frozen native-v2. Job **110561** passed all9 pre-RMS cases (**34,560 exact BF16 outputs**, signed zero equivalent); job **110569** passed the actual whole-O E47 case7 (**3,840 outputs all exactly0.609375**, also passing the unchanged public tolerance). Both result checkers revalidated source/image binding, original CPU truth, input checksums and the actual uploaded tar.

The frozen source is `native-v3/projection.rs` (SHA256 `c89ba9d54baa5b5841426d7bccc6318cd088fa6d60fedc59c88aaf63c768c5be`). The actual whole O image is122,880 bytes, SHA256 `4dde46bfbff2b07c0870f5baf8d5a857b2b0fb7176816f9b0eb417260a671cb4`. The pre-RMS wrapper is a separate diagnostic image calling the complete unchanged production helper; only the whole stress host claims exact whole-image identity.

`oracle_v3.py` reuses the immutable `oracle-src-v2` host/wrapper/reference generator. It explicitly asserts that its generated fixture bytes equal the v2 fixture. `full_stress_v3.py` reuses `stress-src-v1` and the same immutable E47 payload. No new cases, input values, expected outputs or acceptance thresholds were added. See [the original validation description](README.md) for each case and the preserved pre-compile CPU fixture repair.

This verifies the changed consumer/producer mapping on the existing K255/256/4095, row59/60, cluster1919/1920, global-scale and active fourth-term cases. QKV and FFN images are unchanged from E134 in the standard package, so no QKV cache-edge rerun is claimed. The finite tests do not prove arbitrary BF16, every PRNG seed or end-to-end model correctness. Standard-input repeats and extra6 checks remain separate adoption evidence; these diagnostic timings are not performance samples.

```sh
python3 moa-dev/experiments/candidates/o-native-activation-resident/validation/oracle_v3.py --check
python3 moa-dev/experiments/candidates/o-native-activation-resident/validation/full_stress_v3.py --check
```

`verification-summary-v3.json` binds both result records and the original v2 evidence. All prior v2 validation and profile records remain immutable.
