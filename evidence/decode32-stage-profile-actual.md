# Actual matched32 stage profile

Source7a3f90c, a008 ELF d988f30e…; Arena319388 / r880-decode32-stage-profile-submit, PASS23.868s. Same50,331,648 logits and11,811,160,064 reserved KV bytes exact. This is host-observed completed stage timing, not device instruction traces.

Warm32 calls are steps3/6/7; step0 is reported separately as cold. Steps4/5 force64 for both controls despite live7/16, so those are not ordinary B16 costs.

|Stage|Control64 ms|Hybrid32 ms|Mean reduction ms|
|---|---:|---:|---:|
|r64.ffn.core|20.355–20.394|14.815–14.865|5.533|
|r64.ffn.post_norm_residual_gate|2.128–2.161|2.114–2.138|0.014|
|r64.ffn.pre_norm|1.671–1.708|1.677–1.690|0.001|
|r64.full.attention|3.823–3.844|3.794–3.796|0.039|
|r64.full.heads_rope|0.404–0.420|0.406–0.411|0.002|
|r64.full.input_norm|0.277–0.281|0.277–0.287|-0.002|
|r64.full.output|1.468–1.473|1.469–1.472|0.000|
|r64.full.post_attention|0.301–0.301|0.300–0.302|-0.000|
|r64.full.qkv|1.290–1.297|1.290–1.303|-0.003|
|r64.logits.extract|0.000–0.000|0.000–0.000|0.000|
|r64.logits.norm|0.034–0.035|0.034–0.035|-0.000|
|r64.logits.projection|3.395–3.401|2.760–2.762|0.638|
|r64.prepare.decode|0.954–1.307|0.965–1.454|-0.040|
|r64.sliding.attention|28.660–28.851|19.275–19.323|9.490|
|r64.sliding.heads_rope|1.978–1.999|1.967–1.998|0.007|
|r64.sliding.input_norm|1.392–1.416|1.389–1.406|0.005|
|r64.sliding.output|5.368–5.447|5.333–5.397|0.049|
|r64.sliding.post_attention|1.505–1.512|1.520–1.527|-0.016|
|r64.sliding.qkv|6.292–6.329|6.288–6.341|0.002|

Warm32 complete model81.410–82.152 ms →65.774–66.549 ms. Warm17 step1:80.841→64.556 ms. Selection remains separate (~1.0–1.3 ms). First candidate32 cold70.130 ms. Recorded stages account for all but~0.03–0.05 ms of each complete model interval; cache download/validation is outside both.

The accepted reduction is concentrated in sliding attention (~9.5 ms), FFN (~5.55 ms), and LM (~0.64 ms). All remaining64 stages are unchanged within normal variation. Largest individual still-padded operation is QKV (sliding+full~7.6 ms); O is~6.8 ms with a simpler canonical-HBM boundary. Norms total~7.3 ms across five kinds of calls, but each active slice already processes one row, so half-width arithmetic does not imply half their wall time.

Chosen bounded next experiment: exact32-query O inside the existing64-row input/output ABI. It removes half the query contractions and half the16-row VRF pair-add passes, preserving K1024 partials, all BF16 boundaries, both clusters and existing weights. The weight-read cost stays. Even a successful kernel can recover only a few ms from a~6.8 ms stage budget; it is not a multi-fold end-to-end throughput mechanism or a solution to current low-concurrency prompt latency.

Gate: complete sliding/full O ABBA against current ops_output_r64_trf32, all first32 raw partial/output bits, independent F64 dyadic partial/balanced-BF16-tree oracle, provided-model weights, inactive-tail poison preservation. If the inclusive kernel gain is not material, stop this variant. If it passes, integrate only under the existing per-call capacity32 condition and rerun matched all-vocabulary and full-cache whole-model gates.
