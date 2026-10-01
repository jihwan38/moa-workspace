# E149 result: correct but slower, not adopted

Whole141183 passes45 comparisons but O57414 versus contemporary E14742839/42461 (bracket42650) regresses34.616647%. Exact61,440 packet/15MiB geometry, actual Switch/native arithmetic and register gate pass. Performance includes changed ingress/readers/Switch/scheduling; no internal trace attributes the regression exclusively to one physical resource. QKV/FFN images remain unchanged. No repeats, extra6 or boundary NPU were performed. Original boundary packages are preparation/build evidence only. Source/measurements remain frozen.

Adoption rejects ingress-v1; further parameter tuning stops. Next selected candidate is current E142/E147 FFN down30+30 weight ingress/native stages, motivated by current-image141134 up-gather/down-weight overlap. It retains weighted numerical terms and original full60 partial/five12-row consumers. See [decision](decision.json), [whole result](screen-decision.json) and [exact gate](audit-e149-ingress-v1.md).

The following is the preserved selection context.

# E149: O K1024 ingress with Main row/K exchange

Actual start `9b2e7795280201a017ebcd41ad5e773de5656a86`, branch `opt/o-k1024-ingress-switch`. Explicit source E147fb45202 and independent comparison baseline/e147.json. No E148 source changes are inherited.

Current-image E148 diagnostic141134 observed E147 last native starting at weight1-end+2 for every run/cluster. E148's faster quantizer still lost whole performance, so this candidate changes the weight producer/consumer boundary. Each original60row stage is unchanged in size/count. Source slice16g+4a+b owns row4r+b and K1024 segmenta; Main reads quarterq then rowr and InterTranspose4(time0=15) restores slice16g+4a+q and canonical row4r+b before the original K256 native. Source HBM offsets0/60 retain the full120-row pitch. It reads original wire weights; no offline prepack or whole weight transpose is introduced.

HBM/weight DM total15MiB and native2 counts remain. New15MiB in-stream Switch, reader/collect changes and schedule/backpressure may outweigh better ingress. The logical1024B contiguous region may still lower to4×256B descriptor packets. Exhaustively prove actual source/destination coverage, Switch time/slice ordering, original consumer readers/TRF/VRF lifetimes and numerical/postprocessing equivalence before whole NPU. Prior E144 shows compatible Main Switch/native support; prior row4 O added a different output recovery cost and was static-only. This is a coupled local dataflow replacement, not a new arithmetic architecture or a parameter sweep.

Adoption pending compile/gate/whole. Freeze any positive for bounded repeats and original boundary validation. No5percent adoption floor. A source contract failure permits a justified supported repair; a measured regression closes blind tuning. See ingress-v1/recipe.json for explicit three-way decisions and source/control provenance. No performance evidence exists at selection.
