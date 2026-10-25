# Matched actual-model gates

Frozen functional source0bf8c53, buildsource dad0c4a, ELF4ad2b3724f7964b406716b2a60753bddce94a7549b03ff63c5e026ff4728ee85. Root Arena319656/r952 (512history,18.526s) and319655/r953 (1024history,22.682s) both PASS. Total47,710,208 BF16 vocabulary values and13,287,555,072 KV bytes matched exactly. Each run covers18 complete physical owners including the interleaved prompt; raw cache comparison includes untouched storage.

Both arms retain compact metadata, prefix fusion, FFN epilogue, AutomaticR32/O32, media-independent inputs, and original attention math.90 decode vocabularies plus active prompt are compared per run, with greedy/all-banned/None-policy and seeded stochastic checks. Scalar1, B16{2,8,14,16}, R32{17}, resets, fullpage/ring transitions,48 P128 quanta, old-O/canonical-QKV fallback and31-token narrow prompt reuse all pass.

Warm eligible comparisons below exclude the first B16 call. Times are completed model calls excluding retained-logit read/selection. Sequential control/candidate passes are not randomized HTTP throughput measurements.

|History|Step/live|Original model ms|Composed model ms|Original attention+O ms|Composed attention+O ms|
|---:|---|---:|---:|---:|---:|
|512|2/8|40.547|38.825|11.892|10.548|
|512|3/14|40.485|39.015|11.927|10.586|
|512|4/16|40.616|39.037|12.001|10.586|
|512|6/2|39.144|37.926|11.684|10.301|
|512|7/14|40.206|38.622|11.745|10.354|
|512|9/8|39.850|38.505|11.739|10.311|
|1024|2/8|41.024|39.494|11.918|10.572|
|1024|3/14|41.163|39.626|11.972|10.692|
|1024|4/16|41.327|39.668|12.039|10.670|
|1024|6/2|39.149|37.926|11.736|10.314|
|1024|7/14|40.603|39.173|11.810|10.357|
|1024|9/8|40.523|39.276|11.776|10.340|

The scoped stages save about1.2–1.45ms/40sliding layers in warm calls. Warm width14 is40.206–41.163ms original versus38.622–39.626ms composed. Unchanged scalar/R32/fallback calls also vary between passes, so their deltas are not attributed to fusion. The measured micro gain does not imply2.2ms whole-step savings or an HTTP/SLO gain. Proceed with Automatic activation only for pure independent canonical B16 calls using Hsplit O; retain explicit original per-call control and all fallback routes. Group/P512 FFN flags are untouched.
