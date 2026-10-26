# Complete FFN: intended-code correctness PASS, performance HOLD

Actual job **323067** succeeded in **12.82890964s**, exit0. This means the
strict correctness checks completed; the separately reported whole-chain
performance gate **failed in both row orders**. **No model promotion.**
DOWN's prior resource245684>240000 HOLD remains unchanged.

| Row order | Current production E4 | Complete A4 candidate | Candidate slower | Required gain |
| --- | ---: | ---: | ---: | ---: |
| Original512 |1064.6526875us|1108.518625us|4.120211%|at least10% — FAIL|
| True permutation |1069.858875us|1107.611375us|3.528736%|at least10% — FAIL|

Every block averages eight separately completed calls. Both sides restore the
same initial residual outside each sample. All eight measured blocks are retained:

| Order | Block | Arm | Complete us |
| --- | ---: | --- | ---: |
|0|0|A4 candidate|1110.446125|
|0|1|Production E4|1068.004750|
|0|2|Production E4|1061.300625|
|0|3|A4 candidate|1106.591125|
|1|0|A4 candidate|1108.866125|
|1|1|Production E4|1070.124625|
|1|2|Production E4|1069.593125|
|1|3|A4 candidate|1106.356625|

There are32 timed complete calls per arm,64 total. Candidate timing includes
all seven awaits: post-attention norm/residual/preFFN BF16 bridge, fused input A4
quantization/packing, checked raw UP/GATE, original-global/BF16 GeGLU bridge,
fused hidden quantization/packing, checked raw DOWN, and original
postnorm/residual/layer scalar. It includes raw-owner validation/waits and both
canonical BF16 write/reread boundaries. Control is the unchanged q020
`ops_group_ffn_projection_storegroup::sliding_projection_owned` whole graph,
not the old per960 F4 prototype or the integer-coded E4 oracle.

Startup, residual restore, oracle work, uploads/downloads and validation are
excluded from both timers. They are not free adoption costs:

| Reported startup component | ms |
| --- | ---: |
|CPU supplied BF16 reconstruction/per960 rational encoding/audit,176,947,200 values|5860.101283|
|Original production MLP load/native prepack/join|62.686140|
|Candidate **and intended-reference E4** weight upload|221.836340|
|Sum of reported startup intervals|6144.623763|

The upload interval includes diagnostic reference copies and is not a
candidate-only startup measurement. Only provided layer0 was loaded; no48-layer
startup or model capacity result is implied.

Correctness completed196,608,000 BF16 comparison events,6144 sampled independent
i64/F64 envelope checks,4096 exact scale values,19,660,800 packed bytes,
78,645,248 mixed permutation values,599,464,072 directly compared immutable
bytes and226,115,584 post-reuse bytes. These are comparison counts, including
two feedback steps, repeated typed/reference checks and the true row permutation;
they are not distinct model logits. All output boundaries and the final residual
were compared against independent original P128 norm/GeGLU/tail operations,
the unchanged integer quotient oracle and integer-coded E4 projections.
Original production weight/norm immutability additionally checked full-read FNV
fingerprints of394,728,988 bytes; that is explicitly distinct from bytewise checks.

Current-production drift is a separate result. For matched initial synthetic
projected/residual inputs, relativeL2 was **0.5744822058** for joined UP/GATE,
**0.7928396554** for DOWN and **0.8466086464** for the final residual.
Both row orders reproduce these aggregate values. Step1 residual relativeL2
was0.6774843554, but each arm then starts from its independently evolved result;
that number is cumulative feedback drift, not matched-input projection error.
All drift records, changed-value counts and maxima are retained in JSON/raw log.
These fixtures use provided layer0 weights/norms/globals with synthetic finite
activations. **They do not establish task accuracy or full-model quality.**

Declared retained HBM payloads, not observed allocator usage:

| Payload | Bytes |
| --- | ---: |
|Production retained weights/norms/globals|394,728,988|
|Candidate I4 weights/scales|89,210,884|
|Intended-reference E4 weights|177,085,440|
|Candidate scratch|63,901,696|
|Production scratch|49,156,096|
|Subtotal|774,083,104|

The subtotal excludes projected input/table/shift owners, temporary oracle
buffers, host snapshots/file mapping, allocator overhead and kernel DM/TRF.
It is not a48-layer capacity claim. `FURIOSA_OPT_PROFILE=1` was requested and
two profile markers were printed, but `arena-records.json` is empty and the
collected log contains no per-operation trace payload. Thus the regression
cannot be assigned to launches, bridges, packing or another stage from this run.

Provenance: submitted commit `2392c7ba`, build commit `18888930`, functional
native-reference repair `70f4c15f`. All411 submission/build input hashes match.
Frozen ELF6,496,968 bytes, SHA256
`bd51ab5f4c0cf7e20918e022c023c7e42fcbad4fa11b30da382c33297088e61f`.
The18 prior images match their frozen artifacts, including q020's production
graph and loaders; the sole new reference wrapper is covered by this actual gate.
Original dependencies, prior a003 DMA compile failure, resource HOLD and all
negative outcomes remain preserved. No source change, build or extra device run
was performed while preparing this analysis.

Replay from the repository (read-only; stdout JSON):

```bash
python3 experiments/int4-complete-ffn/analyze_actual.py
# Also verify current source and both preserved local ELF copies:
python3 experiments/int4-complete-ffn/analyze_actual.py --verify-local-artifacts
```

Raw files and SHA256 manifest are under `actual-323067/`; `actual-result.json`
contains every timing/startup/drift/sample record, recomputed gates and explicit
scope limitations. Keep this branch on HOLD; do not promote it to model/serving
based on successful process exit or isolated component results.
