# All-live-prefix complete-chain gate: Arena319611

Source315f809; builda004 and immutable ELF provenance are in `build-all.json`. Actual root run is `runs/r930-b16-attention-output-allwidth-submit/arena.log`.

PASS: 17,179,869,184 independently addressed whole-cache bytes and4,063,232 attention/output bytes were exact. All live prefixes1..16, supplied O layers0/6/46, nonidentical rows, physical slot permutation, ring wrap, reset, NaN inactive K/V, untouched neighbors and immutable inputs were checked.

Complete ABBA entries average12 awaited chains. Original is three launches (original one-live-slice commit, original attention, original Hsplit O); candidate is one graph with all original HBM boundaries and arithmetic retained. Metadata uploads and readbacks are excluded identically.

|Live|Original µs (A,A)|Composed µs (B,B)|Mean reduction|
|---:|---|---|---:|
|1|293.543, 292.307|236.079, 233.461|19.85%|
|2|294.564, 292.379|237.066, 235.320|19.52%|
|3|296.033, 294.913|237.990, 236.854|19.65%|
|4|295.602, 293.226|237.031, 234.901|19.85%|
|5|294.754, 292.491|237.296, 235.240|19.53%|
|6|295.527, 292.934|237.208, 235.690|19.64%|
|7|295.483, 293.365|237.304, 234.925|19.80%|
|8|295.458, 292.300|238.027, 235.240|19.48%|
|9|295.558, 294.155|238.331, 235.928|19.58%|
|10|294.995, 293.240|237.493, 236.093|19.49%|
|11|294.468, 293.241|237.926, 235.532|19.44%|
|12|294.989, 293.101|237.984, 236.076|19.39%|
|13|295.401, 292.976|238.958, 236.371|19.21%|
|14|295.495, 293.419|238.275, 235.761|19.51%|
|15|297.969, 295.791|238.225, 236.481|20.05%|
|16|297.585, 294.202|239.011, 236.584|19.63%|

Decision: proceed with a default-off, per-call whole-model comparison. This is a micro result, not HTTP/SLO throughput. Root owns all NPU dispatch. All original stage bodies remain available; no distributed commit is adopted. Actual trace in `actual-319611.json` preserves events by variant/cluster; categories overlap and cannot be summed into wall time.

Actual cluster0 trace corroborates retained work: live14 original3Tasks total463,129cycles versus composed1Task460,589; live16 original466,329 versus composed456,950. Both retain30DMA spans and40TU spans. The task-cycle difference is much smaller than the completed host-chain wall difference, consistent with removing completion/launch boundaries rather than changing KV or contraction work. This is not a measured per-HTTP speedup.
