# M08 pair4-v1 compiled graph independent review

2026-09-27. Read-only review; no compile, host build or Arena execution performed by this reviewer.

Verdict: the first compile implements the proposed row-pair ownership and direct F32 gather. No unexpected copy, restore Switch, arithmetic conversion or graph stage was introduced. The expected gather fragmentation is real and costs 1,020 additional compiler cycles. There remains value in a bounded NPU comparison because the actual HBM address order changed, while the static weight cost model assigns exactly the same duration. This is not evidence of a likely speedup or hardware regression.

## Provenance

Candidate directory: `/home/jihwan38/moa-workspace/moa-dev/experiments/candidates/m08-output-pair4-ingress`.

| Artifact | SHA256 |
|---|---|
| `build-m08-pair4-v1-repro.json` | `733dee4af08fb8ab68def497e19914260589ba29ded236bd7f6d859fa4df322a` |
| `evidence/m08-pair4-v1/graph.json` | `56aceff8e51772ee83bbb3b90ab01be4d37e7d6e3b48fe68e2fea1edeeee245d` |
| `evidence/m08-pair4-v1/schedule.json` | `93e95c18a159ebd62d291b20ce976a29c98fc415c85a090097086c9d9e26bcf7` |
| Frozen `src/device/sliding/projection.rs` | `ae4f6aba8e50dcb6e3118a5a9b25ba105e4dd63a9087148bd716f568232d78bd` |
| Exact O image, 71,168B | `dac23720a6cfbbca3a9bf03387a20e5c66fdef8ceec83775957c7331c82fed71` |

Build record selects E122 `c3553ab87046fb4d59c10149a82517f8440bba27`, the reviewed source override, `ops::sliding_attention_output --exact`, and reports verified container sources / exit0 / compile_only=true. Baseline graph is `artifacts/output-bf16-activation-e47/candidate-graph.json`, whose image SHA `359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b` matches E104/E122 O.

## Actual DMA order

Baseline `DmaCommand#O4` source:

```
[H_1=120:4096, H_120=2:491520, Qs_256=16:256,
 H_240=2:983040, Qs_1=1:256]
```

Candidate actual source:

```
[H_8=60:32768, H_1=2:4096, H_2=2:8192,
 Qs_256=16:256, H_4=2:16384, Qs_1=1:256]
```

Candidate actual destination:

```
[H_8=60:512, H_1=2:256, H_2=2:67108864,
 Qs_256=16:4194304, H_4=2:134217728, Qs_1=1:256]
```

Eight PE bases remain `p*1,966,080B`; local weight allocation remains32,768. Interpreting the emitted stream with last axis fastest gives first packets `0,16384,256,16640,...`, versus baseline `0,983040,256,983296,...`. Thus the proposed H480-per-pair ownership and paired-DMN 16,384B source jump are implemented. Packet size256B, packet count61,440 and logical15MiB are unchanged. The two DMNs already alternated in baseline and still do: this does not introduce a previously absent DMN interleave. Relative bit8 sequence remains0,0,1,1; no bandwidth/channel-occupancy claim follows from the smaller jump alone. A separate checker owns exhaustive generated-descriptor bounds/coverage verification.

## Consumer and gather

Both graphs have29 operators. Exact operator-object comparison changes only:

- `DmaCommand#O4`: weight DMA mapping above.
- `RenegadeCommand#O2`: activation TRF H ownership labels.
- `RenegadeCommand#O7`: Main H ownership/local row factorization.
- `DmaCommand#O11`: F32 gather packetization.

The actual Main O7 retains the baseline fetch sequencer `0+[1920:16,16:1]`, BF16 fetch network, same lookup table T12, `GReduceAddF ratio=16`, Float32 transpose2rows→1row, `commit_in_size:8B (=2EA)` and commit projection `0+[60:2,2:1]`. The only row changes are slice `[H_480=4,H_2=4,Qs_256=16]` and local `[H_8=60,H_1=2,...]`. Thus contiguous16 K-column reduction, paired F32 commit and the intended local120 rows remain.

Actual gather O11:

```
src: [H_8=60:8, H_2=2:67108864, H_4=2:134217728, packet1:8]
dst: [H_8=60:32, H_2=2:8, H_4=2:16, packet1:8]
```

Eight descriptors mean8×60×2×2=1,920 packets of8B, vs baseline8×2×2=32 packets of480B. Logical payload remains15,360B. Destination H480 owners remain cluster0/slices0..7. Source F32 allocation63,488 and destination17,152 are unchanged. There is no additional intermediate tensor or restore Switch, but small-packet service cost increases.

All channel-scale, RMS, residual and HBM-store operator objects are exactly equal to baseline. Only five tensors change mapping metadata (T6,T7,T9,T13,T14); all other tensor objects are equal. This supports preservation of the intended numerical operations and rounding boundaries. Compile/graph equality does not replace NPU correctness validation.

## Static cost and NPU decision

| Critical interval | Baseline | M08 | Difference |
|---|---:|---:|---:|
| Weight DMA | 13,383 | 13,383 |0|
| Main lookup/dot/reduce/commit |5,063|5,063|0|
| F32 gather |470|1,490|+1,020|
| Whole O makespan |27,239|28,259|+1,020 (+3.7446%)|

The same weight→Main whole-tensor dependency remains: weight ends16,157 and Main begins16,157 in both compiler schedules. M08 does not enable early partial-weight consumption or overlap that stage with its own dot. Ingress improvement, if any, must come from actual transfer behavior under the changed address order. All downstream operations are shifted by the extra gather time with their own durations unchanged.

There is no positive compiler speedup to advertise. However, static equal weight duration does not measure whether the new HBM sequence shortens or lengthens the observed26,736-cycle hardware weight interval. The baseline weight/dot/gather static-to-NPU relations also differ, so neither multiplying all costs by one ratio nor carrying the3.7446% prediction over to hardware is justified.

Proceed with a bounded NPU diagnostic/whole-O screen once the generated-descriptor checker and package identity gates pass. Compare same-cluster weight completion, dot start, gather and tail, and retain contemporary complete-O controls. Actual ingress savings must exceed the added gather and any synchronization effects. A weight-only result is not a whole-kernel gain; no NPU correctness or performance result exists at this review point. If the whole screen gives no material gain, the current candidate can close without a tile/row-layout sweep.
