The fixed provided-gamma policy is **not promoted**. Job 323281 succeeded in 26.329942824 seconds and all compensation/intended-code prerequisites passed, but its final residual drift increased in both matched-feedback steps. This is one layer 0 diagnostic with provided weights/norms and synthetic 512-row inputs, not a model-quality or performance test.

Source `e6e45d96` / functional `52d821e2` used the original locked dependencies. The submitted ELF is 6,584,096 bytes, SHA256 `a71cf0d522cab1d731d3a95d84fabfa17ae4052bef965382f7250fe13781dbb5`. All 423 source/config inputs match the fresh captured build; all 19 embedded device images match both actual 323213 and the new empty output directory. Submission status recorded only five untracked build-evidence paths, with no source diff. [Build bindings](build-result.json), [raw submission](actual-323281/run.json), [terminal status](actual-323281/arena-status.json), [raw log](actual-323281/arena.log).

Every one of 117,964,800 transformed UP/GATE values passed the normal-or-zero BF16 range/exact inverse gate. All 7,864,320 actual prefix values passed signed-bit/F64 inverse compensation. The sampled 15,728,640 unquantized product pairs (4,096 dots) matched F64 terms/same-order sums and normal-or-zero F32 product bits before quantized drift was interpreted. Intended dense-code/reference parity covered 393,216,000 BF16 comparisons plus 12,288 F64-envelope samples. Both row orders passed full permutation, reuse and immutable-owner checks. Counter totals are 157,368,320 heterogeneous permutation elements, 1,198,677,136 immutable compared bytes and 771,375,104 reuse/control-scratch compared bytes; repeated comparisons are not unique allocations.

All three arms received the same original-production incoming residual at both steps. Step 1 used original-production feedback, never a candidate's state supplied only to its control. The table shows relative L2 in the original comparison coordinates; both row orders give the same changed counts/maxabs and L2 within 2e-12.

| Comparison | Unbalanced grouped A4 | Gamma-balanced grouped A4 |
|---|---:|---:|
| UP weight reconstruction, restored original coordinates |0.203050|0.630733|
| GATE weight reconstruction, restored original coordinates |0.224123|0.541710|
| Prefix activation reconstruction, step 0, restored original coordinates |0.427418|0.534348|
| Prefix activation reconstruction, step 1, restored original coordinates |0.346974|0.338761|
| Final residual vs current q020 E4 graph, step 0 |0.423881|0.662802|
| Final residual vs current q020 E4 graph, step 1 |0.374813|0.459115|

The transformed-coordinate activation L2 is 0.199607/0.174906 in steps 0/1. Those smaller numbers use a different reference scale and do not establish improved original-coordinate reconstruction. Step 1 restored activation L2 improves slightly, but its max error rises 20.5→35.75 and the final residual worsens. UP/GATE reconstructed zeros rise 18,632,096→37,488,784 and 23,160,065→37,873,795, respectively. Scalar reconstruction uses RN32(code×scale) and F64 SSE with exact dyadic inverse compensation; it is not an exact dot-error decomposition.

The reported 13,076.412607 ms CPU interval includes original/balanced decoding, complete inverse auditing, both encoders and metrics. It excludes serving startup/throughput attribution. Arena duration includes device work, CPU oracles and readback; no complete-chain timing gate was run. No policy change, model activation, task-quality improvement or global transform equivalence follows from the prerequisite PASS. The previous grouped-complete timing HOLD remains intact.

Replay from this worktree without executing an ELF, build, or network command:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 experiments/int4-channel-balance-diagnostic/analyze_actual.py > /tmp/gamma-actual-replay.json
cmp /tmp/gamma-actual-replay.json experiments/int4-channel-balance-diagnostic/actual-result.json
```

The analyzer pins every copied raw-record hash, verifies the 423-input source/ELF/image chain, consumes all 124 log lines in their expected order, checks four matched-step markers/counts and compares row-order aggregates. It requires the retained build/submitted ELF and fresh image paths named in [build-result.json](build-result.json), reads them without mutation, and produces [actual-result.json](actual-result.json). Log SHA256: `dbd521efe4fd15bb68e74d954e28f271ac3223c6362e0939ccb2959416a242d5`. Independent read-only review by `/root/e2e_evidence` corroborated raw hashes, counts and the limited no-promotion interpretation.
