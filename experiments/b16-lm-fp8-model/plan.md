Default-off integration from current c079394d60a2 plus frozen LM micro3a877b1e.
One immutable E4 table/scales owned by Prefill, original Weights.embedding_table
retained. Explicit startup prepare allocates once; source constant remains false.
A per-call diagnostic dispatch chooses native or original B16 LM, then delegates
all other dispatch to Automatic. No persistent mutable precision mode, cache,
Worker, token-selection, prompt/scalar/R32/R64 or wide LM changes.

Reuse Prefill's already-owned native_q_input/native_qkv_scale only after final
layer+RMS. Complete both launches before existing extraction or retained-logit
return. Active prompt hidden states live elsewhere; subsequent layers overwrite
these existing temporary tensors. Missing-shadow requests fail before forward.

Model probe: real512/1024 teacher-forced histories on17 owners in disjoint original/
reversed canonical allocations; same explicit final-RMS/dense/cache arithmetic.
B16 widths2/8/14/16 report approximation metrics and selections; scalar1/R32-17,
active P128 interleaving and final full reserved-KV bytes remain exact. No generated
candidate token is fed back. Original selection APIs validate each variant's own
logits, including banned/all-banned/None and seeded stochastic differences reported.
Prepare current joined projection FFN and group/greedy scratch before measuring
shadow allocation; log actual allocator state through histories/warmups. This is
full current-model capacity, unlike the earlier standalone measurement.
No default activation or public quality claim until root runs gates.
