# E138 native-v1 whole-QKV readiness diagnostic

One frozen-image allocation111237 passed **81 comparisons**, including original whole E137 before/candidate/after and QKV-only diagnostic info/trace. Whole QKV again regressed:104,613→118,231→104,852, **+12.8885%** against bracket104,732.5. Diagnostic medians were E137104,139/info and106,538/trace; candidate118,038/info and120,981/trace. These diagnostic timings are not adoption evidence.

Both original kernels retain all instructions, constants, addresses and I/O. Only existing profile-name bytes change in the diagnostic images. All three images in each diagnostic host bind to their original sources; standard hosts are unchanged. All76 control and90 candidate executable operations match marker order/kind. Four source-supported cluster0-only DMA operations are absent on cluster1: rope cos/sin gathers and K/V cache scatters. All observations use their own cluster's task clock.

| Candidate landmark | Median begin | Median end | Median interval |
|---|---:|---:|---:|
| preRMS apply |7,763.5|8,728.5|965|
| H480→consumer H960 typed input |8,729.5|35,255.5|26,638|
| K4 max |36,204.5|36,617.5|412|
| Last term quantizer |51,815|53,389|1,574|
| Shared activation TRF |53,390|54,683|1,293|
| V weight ingress |3,382.5|22,043|18,413|
| First V native |54,684|56,916|2,237|
| Q weight ingress |44,136|79,049.5|35,109.5|
| Q native |79,051.5|82,491.5|3,319|
| K weight ingress |87,609|106,352|19,209|
| K native |106,354|108,581|2,224|

Endpoint medians need not subtract to the median interval. Six run/cluster observations are not six allocations. Within each observation, preRMS end→input DMA start and TRF end→first V native are exactly1 cycle. V weight is ready33,458 cycles before native start (paired median). Q/K native starts are exactly2 cycles after their own weight ingress ends. Thus activation preparation delays the first native consumer, while later Q/K consumers are weight-ready limited. This selects a concrete input-routing experiment; it does not imply all input time is recoverable or that weight/head costs disappear.

The selected alternative is documented in `../conditional-input-routing.md`: type only120KiB BF16 column/ring seeds instead of960KiB consumer input, then exact ring32 Main broadcast back into the same512-slice consumer layout. Quantizer, four factors, K4 weight mapping, F32 reductions and head arithmetic remain fixed. Main Switch and seed storage/read/write are additional costs. Its complete NPU result is required; no sum of overlapping intervals is a speedup prediction.

Historical tool repairs are explicit. `package-profile-v1.json`/`package.py` preserve the first visibility metadata omission; `package_v2.py` fixes the cluster0-only list without recompiling the hosts. Separately `collector-correction.json` reproduces the frozen original `screen.py` parser's unbound `by_label` variable using the actual raw log. `collect_v2.py` changes only the operation-label list expression; the actual uploaded source/runtime and kernel bytes are unchanged. No extra allocation was used to repair parsing.

```sh
python3 moa-dev/experiments/candidates/qkv-native-consumer-local/profile/analyze.py e138-readiness-v1 111237 --check
python3 moa-dev/experiments/candidates/qkv-native-consumer-local/profile/verify.py e138-readiness-v1 111237 --check
```

The measured prototype is not adopted; no row/K/term sweep follows. The profile selects only the bounded native-v2 routing repair and does not close the native QKV family.
