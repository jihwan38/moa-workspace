# E137 finite O correctness checks

The frozen `native-v2` production projection is tested at two boundaries. These are correctness diagnostics, excluded from performance measurements and official submissions. The production sources contain no case-dependent branches. The host generates weights and checks their checksums before launching the same compiled kernel for each input.

## Pre-RMS production helper

`oracle_v2.py` binds every original source file to `build-e137-native-v2-repro.json`. It appends a diagnostic module to `lib.rs`; the wrapper calls the complete production `project_output` and copies its BF16 output to HBM before RMSNorm/residual. The entire production `projection.rs` remains byte-identical. The separate diagnostic image is not claimed identical to the whole-O image.

`oracle-src-v2/generate.py` derives expected outputs from stored BF16 inputs and original decoded E4 weights using F64 dot/channel scale and BF16 round-to-nearest-even. No transformed four-term E4 input is used as an oracle. All chosen full dot and scaled results are asserted exactly representable in F32; this avoids attributing a different reference rounding to the candidate. The host checks BF16 bits, permits either signed zero, rejects nonfinite values, and pre-fills output with NaNs to expose missing writes.

Nine cases cover 34,560 outputs:

| Case | Purpose |
|---|---|
| 0 | Signed zero and output completeness |
| 1–3 | Isolated K255, K256, K4095 |
| 4 | Mixed signs at K0/255/256/511/512/2047/2048/4095 |
| 5 | Every 120-row group, local rows0/58/59/60/61/118/119, including both 60-row stages and cluster boundary1919/1920 |
| 6 | E47 tiny correlated cancellation: each K256 has +1,−1 and254 values2^-22; original weights1,1,then240 |
| 7 | Channel scale before first BF16; premature BF16 dot casting changes1920 expected outputs |
| 8 | One K256 shard establishes global amax; all other shards contain129×2^-29, with signed row/cluster weight patterns and power-of-two channel scales |

At global scale2^-7, case8's tiny normalized value129×2^-22 has four terms `[0,0,128,32]` under restoration divisors `[1,16,2^22,2^27]`. The fourth term must contribute. Its global-scale and fourth-term checks complement cases1–5; it does not by itself identify every possible K permutation. Case8 is a finite BF16 diagnostic, not an observed public PRNG seed or a claim that every diagnostic component lies on the public PRNG grid.

The initial harness preparation (`oracle.py`, `oracle-src-v1`, `oracle-build-v1.json`, `oracle-evidence-v1`) failed its CPU assertion before any compile or NPU submission: case8's dot was exact F32 but multiplication by inherited non-power-of-two row scales was not always exact F32. Harness v2 changes only case8's channel scales to powers of two. The failed source and log remain immutable. This is a test reference construction repair, not a kernel failure or precision workaround in production.

## One actual whole-O regression case

`full_stress.py` builds an additional host around unchanged production sources, without a device wrapper. Its sole embedded `ops::sliding_attention_output` image must equal the frozen whole-O exact image. `stress-src-v1/probe_output.rs` selects only legacy E47 case7 from the immutable E47 payload, preserving the coordinate weight rule, input checksums and original public tolerance `abs(got-want) <= 0.05 + 0.01*abs(want)`.

The immutable `boundary.safetensors` payload is379,768 bytes, SHA256 `35df91a34ca00b077b35327c513bb36783355363855e214674ddaa04a47ec907`. `stress-src-v1/cpu_reference.py` re-evaluates only this case using original `scaled_projection`, RMSNorm and BF16 residual. The current and historical `scaled_projection`, `gen_sliding_attention_output`, `rms_norm` functions are byte-identical as extracted from their AST; reference module and PRNG files retain their original hashes.

For each K256 shard, x is `[+1,−1,254×2^-22]`, weights are `[1,1,254×240]`; channel scale2^-10, gamma0.5, residual0.5. All3840 rows have raw dot0.2325439453125, projected BF16 value0.0002269744873046875 and final reference0.609375 (`0x3f1c`). This is a structured component-range regression case, not an observed PRNG seed. Re-running all eight legacy whole-O cases is outside this bounded check.

## Reproduction and scope

Builds use `/tmp/moa-m03-validation-probe/target` sequentially. Both submissions require the candidate's passed graph gate. Build records bind source bytes, separate images, hosts, fixtures, CPU reference and logs. Result checks also re-read the uploaded tar and compare every member with the prepared package.

```sh
python3 moa-dev/experiments/candidates/o-native-activation-resident/validation/oracle_v2.py verify
python3 moa-dev/experiments/candidates/o-native-activation-resident/validation/oracle_v2.py --check
python3 moa-dev/experiments/candidates/o-native-activation-resident/validation/full_stress.py verify
python3 moa-dev/experiments/candidates/o-native-activation-resident/validation/full_stress.py --check
```

The standard package independently verifies that QKV and FFN images are unchanged from E134. No QKV cache-edge run is inherited from E136 or newly claimed here. These finite O tests establish their stated input/boundary coverage; they do not prove arbitrary BF16 correctness, all public seeds, or end-to-end model correctness.

NPU job **110458** passed all nine pre-RMS cases: **34,560 exact BF16 outputs**. Job **110465** passed the one actual whole-O case: **3,840 outputs**, all exactly 0.609375 in F32 as well as passing the original public tolerance. Both uploaded archives and source/image/CPU provenance passed the `--check` commands above. Results are frozen in `oracle-results-v2.json`, `stress-results-v1.json`, and the digest-bound `verification-summary.json`.

The independent whole performance screen regressed; these correctness passes do not override that result or justify additional parameter exploration. The tests preserve evidence that the selected four-term precision/ownership combination works on these cases, separate from its measured complete-path cost.
