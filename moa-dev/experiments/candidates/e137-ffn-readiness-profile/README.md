# E139 current FFN readiness diagnostic

One Arena allocation, **111124**, passed all **36 output comparisons** with current E137 images. Standard whole-FFN medians before/after were **314,052 /314,889 cycles**; diagnostic info/trace medians were **313,350 /313,883**. The trace therefore measures the current weighted-native-down path with small observed host/tracing differences. It is neither a new performance candidate nor an official submission.

The diagnostic started at `d6e6c62e6ae80c9b39b192c4a318a0230e79ee20`, branch `diag/e137-ffn-readiness`. Source/control is E137 `334a7059a82252a310436c6d7e02f8ba577e7f51`. `package.py::bindings` verifies the full46-source adopted manifest/tag and current source bytes (the tracked lockfile supplies the ignored worktree Cargo.lock), actual standard host/fixture/runner and all three images. E137 FFN is byte-identical to E134; only the unrelated sliding projection file differs between their complete manifests. Thus E134 weighted-v3's frozen graph/schedule is reused for FFN, while both whole controls use the **current E137 standard host**.

Only the diagnostic host and existing FFN profile-name bytes change. Task code, constants, IO, slots, marker count and marker positions remain unchanged. All137 executable operation markers match schedule kinds and order. The final graph and resource schedule have different ID namespaces: no numerical ID join between them is assumed. `profile-analysis.json` includes the schedule's actual input/output tensor IDs, storage shapes/lifetimes and source bodies; the original graph is separately digest-bound. For example, native-down schedule637 consumes weight213 from390, activationTRF297 from633, coefficients294 from625, and table257 from502.

The four stages were whole before → diagnostic info → diagnostic trace → whole after, all using the same three original standard inputs. Three inputs and two clusters supply six within-clock trace observations, not six independent allocations. All times below are per-cluster Task-relative medians; median endpoints need not subtract to the median duration. Full raw logs, uploaded archive hashes and samples are preserved.

| Current-path event | Schedule index | Begin→end median cycles |
|---|---:|---:|
| Pre-RMS canonical output gather |141|14,371→15,674|
| Up packed-weight HBM |120|8,161→55,765.5|
| Up FP8 materialization |153|55,775.5→75,898.5|
| Up input gather |159|56,192.5→68,930.5|
| Up amax |180|75,899.5→78,192.5|
| Up packed terms leader transfer |299|93,820→98,868|
| Up ring32 term broadcast |305|98,871→122,279|
| Up activation TRF |324|122,280→125,492|
| First /last up native |328 /376|125,493→128,891 /142,635→146,004|
| Gate packed-weight HBM |307|99,231→148,042|
| Gate FP8 materialization |385|148,052→168,176|
| First /last gate native |422 /469|169,017→172,337 /186,082→189,448|
| Gate local gather marker |475|189,449→227,228|
| Down packed-weight HBM |390|148,472→226,550|
| GeGLU output |499|229,934→230,444|
| Down canonical L7680 gather |506|230,445→231,905|
| Down canonical L7680 amax |509|231,907→237,150|
| Down third weighted term |589|243,667→244,641|
| Last term-copy commit |594|244,642→245,377|
| Compact term gather |603|245,393→246,843|
| Down leader transfer |609|246,845→253,429|
| Down ring32 broadcast |617|253,431→265,322|
| Down activation TRF |633|265,323→267,096|
| Down native +weighted term reduction/partial write |637|267,097→288,105|
| Down blockscale HBM |626|257,407→277,728|
| First /last retained partial scale/reduce |655 /707|288,506→290,735 /301,323→303,557|
| Cluster finish/first BF16 |730|305,392→306,237|
| Final output store |774|312,080→312,921|

`prepare_down_fp8_split` first reduces amax across each cluster's canonical L7680 in a single-slice vector path. Its term quantization then operates on480-element shards. The amax, shard copy, scale/coefficient preparation, three weighted terms, explicit term copies, compact gather, leader replication and ring32 broadcast are distinct operations; this is not the older E128 BF16-down path.

The first up native starts **1 cycle after its activation TRF** in every run/cluster, while decoded up weight is ready49,655 cycles earlier (median). Up term distribution is therefore a material readiness boundary too. Pre-RMS finishing early does not imply its prepared terms can be consumed early; weight/TDMA/Main work and replication intervene.

Down native also starts **1 cycle after its activation TRF** in every observation. Down packed weights, table and restoration coefficients are ready40,922 /29,977 /9,360 cycles earlier, respectively. GeGLU completion→TRF readiness is a **36,680-cycle median elapsed window**. This supports evaluating consumer-local down preparation; it does not support treating the window as a guaranteed saving. At roughly11.7% of the314k whole-FFN runtime, it is not a complete solution to a roughly31% FFN reduction target. New typed-layout transfers, more active quantization/storage, shared resources and downstream blockscale ingress can change the schedule.

Gate gather's marker starts immediately after the last gate native, while down weight HBM is still in flight. Its output becomes ready a median **678 cycles after down weight completion**. The long gather envelope includes waits/overlap; neither a pure bus stall nor an independent37k removable cost is established. Likewise, retained down partial processing is a separate cost boundary: native writes the F32 partials, five subsequent scale/reduction streams consume them, and the final cluster reduction/epilogue remains. No durations of overlapping operations are summed.

The three decisions are recorded in `decision.json`: no kernel adoption applies to this diagnostic; stop after the one validated allocation; present consumer-local down quantization for root's exact layout/precision/cost review. It must preserve the actual per-cluster L7680 scale and weighted `[s,s/16,s/256]` contract and prove L1920 consumer ownership,7680B F32 residual VRF plus scale/coefficient lifetimes. E135's unresolved runtime liveness receives no arbitrary padding or context retry from this result.

`binding-attempt-v1.json` and `package-binding-v1.py` preserve one pre-build tool failure: the new worktree lacked the ignored crate Cargo.lock. The repaired binder checks the baseline's tracked lock bytes; the original measured crate used for compilation contains the exact lockfile. No kernel compilation or NPU run occurred in that failed attempt.

Reproduction:

```sh
python3 moa-dev/experiments/candidates/e137-ffn-readiness-profile/verify.py e139-ffn-readiness-v1 111124 --check
python3 moa-dev/experiments/candidates/e137-ffn-readiness-profile/analyze.py e139-ffn-readiness-v1 111124 --check
```

`analysis-summary.json` is the compact result. `profile-analysis.json` contains full operation observations and source/tensor links. `verification.json` independently checks actual uploaded bytes, the original three images, diagnostic name-only replacement, source contracts and all36 comparisons. Both original and labelled hosts/images, compiled input sources and the uploaded archive remain in the shared ignored artifacts directory. No runtime source was edited.
