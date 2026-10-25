# P128 old-history native E4 QK resource bound

Base: `4e024028`, current c079 model arithmetic. Root owns this isolated tree
and reuses the idle r64-server target. No serving activation or device test.

The current `ops_prefill_head_attn_commit::sliding_fused_p128` prepares BF16
headed queries, attends to immutable old and current keys, then commits current
K/V. Its old-history QK uses BF16 packets of32 for Ds256, with K in TRF and
128 local query rows. The current chunk QK, BF16 PV, score rounding, softmax,
row denominator, residual cache ordering and full head preparation remain intact.

The experimental bound changes only old-history Q/K to BF16→F32 fetch, E4 cast and real DM materialization
and native64-wide E4 contraction. It deliberately omits adaptive scaling and
therefore is not a candidate numerical algorithm: plain casts can saturate or
underflow. The initial direct BF16→E4 Fetch Adapter call is unsupported (r1206); its compiler failure is retained. Query cast happens on the four local rows before the original head broadcast. Both inputs still read the existing BF16 cache/storage. This tests
whether cheaper old-history QK has enough complete-chain headroom to justify
a separate, correctly scaled implementation and its actual quality gates.

Compare complete fused head/attention/commit resource schedules in the same
source, including cast, redistribution, original mask/softmax/PV and cache costs.
At least10% complete-chain gain with no new DramStack is required before any
scaled algorithm work. This is a compiler screening criterion, not a hardware
speed result or a universal theoretical bound. A K-shadow storage change could
remove additional DMA but is outside this experiment.

Prior R64 native QK and its slower head-quantization-before-broadcast remain
separate HOLD experiments. This route has128 local query rows and a distinct
current production fused chain. No old R64 quality result is inherited.
