The bridge source is `381fc023`, based on `cd331f3e`. The production comparison is
that base's `ops_group_ffn_projection_storegroup::{sliding_projection_owned,
full_projection_owned}`. Neither control graph nor any normalization helper was
changed. `verify_source.py` proves full-file equality for those controls and
function-body equality for all FFN stages; only `global_dm` and
`geglu_tile_into` became `pub(crate)`.

The new disconnected APIs in `ops_int4_ffn_bridges` are:

| Entry | Inputs after `device` | Mutable outputs |
| --- | --- | --- |
| `prefix_sliding_bf16` | Four BF16 `ops_output_p128_h64_blocked::Output`; post-attention/pre-FFN BF16 `[H]` weights | Blocked `ops_norm_blocked_residual::Residual`; canonical BF16 `InputBf16=[P512,H]` |
| `prefix_full_bf16` | Four BF16 `[P128,H]`; same norm weights | Same residual and input-BF16 output |
| `geglu_bf16` | BF16 `ops_ffn_projection_clusters::JoinedOutput`; existing F32 `[1]` up/gate globals | Canonical BF16 `HiddenBf16=[P512,L]` |

The positional argument order is `(device,a,b,c,d,post,pre,residual,output)`
for either prefix and `(device,joined,up_global,gate_global,output)` for GeGLU.
Output owners must be distinct from every input and from the mutable residual.
Every call must finish before its mutable buffers are reused. The prefix writes
all 512 rows, including initialized padding rows, and updates the FFN residual.
It does not touch KV, owner state, cursors or layer gates.

The future complete path is:

1. Prefix produces BF16 directly before the current E4 quantizer. No E4
   quantization or reconstruction occurs.
2. The validated H rational quantizer/packer consumes canonical BF16 `[P512,H]`.
3. Checked raw UP/GATE writes the existing joined BF16 output layout. Its per960
   scales are already applied before the reduction. Do not apply the native-E4
   row-dequant factors again.
4. GeGLU consumes those same dots, applies the loader's reciprocal globals,
   rounds each scaled UP/GATE to BF16, executes the original F32 Erf/product/
   DivF2 sequence, and rounds the hidden result to BF16.
5. The validated L rational quantizer/packer consumes canonical `[P512,L]`.
6. Checked raw DOWN produces the original blocked H240 output. Apply its global
   at the original post-first-BF16 boundary in that producer, exactly once.
7. `ops_norm_blocked_residual::post_feedforward(device,output,post_ffn,
   layer_gate,residual)` supplies the original postnorm → BF16 residual add →
   BF16 layer gate tail. This public entry already exists and was not changed.

The original globals come from `host/load.rs::reciprocal_global_scale`; they are
already reciprocals. There is no new inversion or calibration in either bridge.

For an independent prefix oracle, copy initial residuals into separate owners,
use `ops_norm_blocked_residual::unpack_four`, and for sliding projections use
`ops_output_p128_h64_blocked::restore_four`. For each quarter execute the original
`ops_norm_wide::post_attention_p128`, then `ops_norm_wide::rmsnorm_p128` on the
updated residual. Compare all four residuals and all normalized BF16 rows after
canonical host indexing. These entries expose the pre-quantization boundary;
comparing only downstream E4 codes would hide possible BF16 differences.

There is no existing public GeGLU entry that exports all pre-quantization BF16
hidden rows. A diagnostic reference can extract the frozen private
`scale_tile`/`geglu_tile_into` and pre-amax portion of
`ops_native_fp8_p512_blocks_sharded::geglu_quantize`, using independently owned
UP/GATE plane buffers and a separately indexed canonical export. Record donor
body hashes. This keeps the old per-call global loads and separate inputs,
independent of the new joined view/row-store path, while using the same device
arithmetic. A CPU F64 sample is an additional numerical check, not an exact CPU
implementation of SDK Erf/DivF. Compare on the *same joined dot values*, since
I4 and production E4 projection results are intentionally different.

For the intended complete-I4 oracle, dense E4 code expansion plus the same
per960 scales before inter-slice reduction remains the reference for raw GEMM.
The current production E4 full graph is a separate drift/timing arm, not a
bit-exact reference for newly quantized activations or weights. Immutable weights,
globals/norms and separate poisoned outputs need checking after reuse/timing.

All new movement belongs in complete timing. Canonical prefix write is
3,932,160 B; GeGLU hidden write is 15,728,640 B. Their subsequent quantizer reads
add the same bytes: 39,321,600 B total write/read boundary traffic, apart from
original joined UP/GATE reads (31,457,280 B), residual I/O and other unchanged
operations. Both joins are real DM transfers. The GeGLU row staging uses 64 live
leaders in 256 physical slice positions per cluster, so the compiler's padded
allocation and lifetime must be counted rather than just live output bytes.

The source statements retain output-before-residual persistence ordering.
The resource scheduler may issue the disjoint residual store earlier; no caller
observes an intermediate state within the awaited launch. Do not add output/
residual aliases or infer a DMA ordering fence from source statement order.

This handoff authorizes no device execution, serving activation, complete-FFN
performance claim or model-quality claim. Those remain separate root gates.
