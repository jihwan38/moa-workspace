Current measured boundary: P128 QKV materializes one shared quantized input.
Actual318102 gives complete slidingQKV503us andfullQK403us with2.4–2.5%
synthetic activation error. Full-model/public quality gates are still pending.

No QKV-to-FFN quantization reuse is legal: WidePrefill::sliding/full first
normalize the incoming residual with layer.input_norm. After attention/O,
post_attention updates x; ffn then normalizes this new x with pre_feedforward_norm.
The two quantizers consume different values and different norm weights.

The next concrete batching mechanism is original Q/K/V weight retention across
four independent128-row groups. Current GroupPrefill::advance_layer calls each
WidePrefill::sliding/full completely (include_ffn=false) before moving to the
next request, so it reloads Q/K/V weights pergroup. Grouping only FFN cannot
remove these reloads. A grouped QKV route must split that attention phase into:
1. Perrequest input norm; join four row ranges and quantize once byrow.
2. Shared P512 Q/K/V core with original weights loaded once, four TRF64 row
   tiles percluster; independently compute every row.
3. Perrequest heads/RoPE/causalattention/O/residual, restoring its own blocks.
4. Existing proposed grouped FFN or four validated P128 FFNs, then advance all
   request cursors together. Preserve distinct pending allocations and x states.

Dense-agent ABI856ab1c fixes ops_batch_layouts::P512H960 as
[P512/256,H/960,P512%256,H%960]. A64-row tile is61,440 contiguous FP8 bytes.
P512L960 andP512L240 similarly support its FFN experiment. Quantizer/layout
correctness and the blocked P512 FFN performance gate remain unmeasured at the
time this note was written; original P512 FFN2.865ms merely matches four current
P128 FFNs(~2.884ms) before joins, so existing GroupPrefill remains held.

For source-level placement accounting, Qs original weight has15,728,640 E4
bytes. Each P128 core places identical weights in two token clusters; four
separate cores perform four such placements. A P512 core can retain each
cluster's shard across four TRF64 tiles, eliminating three weight placements.
Activation placements still scale with512 rows. These are logical data-layout
counts, not measured HBM traffic: compiler TDMA multicast/transfer scheduling
may change actual bytes/time. Do not infer an end-to-end speedup from them.

The decisive next gate is one P512 Qs core against four already-validated P128
prequantized cores on identical512 rows, plus independent quantized F64/layout
checks and complete quantize+projection+split timing. Compare against native
P128, not the slower BF16 baseline. Only after a net gain and an accepted native
QKV model should Q/K/V expansion and grouped host phase factoring proceed.

An alternative exact-layout implementation accepts four existing P128 input
and output tensors in one kernel and holds weight DM across four core calls.
This avoids explicit row joins/splits but retains strided HBM activation reads;
the blocked-input candidate targets those reads as well. Do not implement both
as a parameter sweep; first inspect the dense agent's blocked IO attribution.
