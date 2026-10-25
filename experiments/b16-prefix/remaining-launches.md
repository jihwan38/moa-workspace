# Remaining B16 host boundaries after prefix composition

Source: this export, based on6b55cb0. Actual profile: r876/319380, pass1,
warmed step7 with10 live requests and epilogue enabled.

`Prefill::decode_sliding` has five sequential launches per sliding layer:
1. composed input RMS, activation quantization, QKV and heads/RoPE;
2. `PoolOffsets::sliding_canonical`, one VALID-specific scatter;
3. `ops_request_sliding_b16_rc64::sliding_attention_b16`;
4. `ops_output_b16_hsplit::project_sliding_output_batch`;
5. `post_attention_ffn_native`, the already-composed epilogue.

Across40 sliding layers, prefix6.245ms, commit+attention8.875ms, O2.863ms.
All48 epilogues together14.116ms; each is already one device graph. The stage
timer does not separate the commit from attention or compute from launch delay.
It would be incorrect to call the whole8.875ms host overhead.

A bounded next composition is attention -> O -> epilogue, retaining every HBM
stage and leaving the current-token commit before it. This removes80 waits per
decode step without16 VALID variants. Prefix removed144 waits and measured
mean2.639ms across six warm paired diagnostic steps, suggesting roughly1.5ms
for80 only if that boundary cost transfers. This is an unmeasured hypothesis,
not an expected serving or throughput gain. A larger whole-layer graph may
reduce more boundaries but has greater compiler/resource risk.

Full attention remains commit -> first/next page loop -> divide -> O -> epilogue.
Its page count is request-dependent; avoid changing page order or masked
recurrence merely to reduce calls.

Separate host inefficiency: decode_batch allocates/uploads row,kv_s,kv_f for
every row even when pooled metadata is present. These three tensors are used
only in the unpooled fallback branches. Root ab46131 addresses this independently.
The profile also includes per-request logits extraction and offset uploads,
which native retained Batch16 serving skips; do not claim all prepare.decode
or logits.extract time as production savings.

No further composition implementation is included in this export.
