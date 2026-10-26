# Fixed metadata store repair

Plan `0d4ac55d`; functional source `f6ad9c83` (prior grouped source55a55aff retained).
The only graph delta is a real compact-scalar DM gather before each offset/scale
HBM store. All other kernel bytes mechanically revert to the prior source;
probe changes only its imported kernel module. Independent source and resource
review passed. Host check passed10.73s.

| Complete static graph | Prior grouped | Group-leader stores |
|---|---:|---:|
| H4 |60292|29069|
| L16 |74636|67271|

H metadata changes each17823cycle HBM store to716DM+451HBM; L changes each8457
to609DM+477HBM. All added movement is included. Each metadata staging owner is
512KiB physical,8192/32768B live. The two owner lifetimes do not overlap. Both
graphs have only the six explicit DramIO arguments and no DramStack; the HBM
offset store really completes before the dependent table gather.

The complete L decrease7365 is smaller than the sum of changed operation spans
because of overlap. Current H/L table gathers7886/28521 and input loads4783/15950
remain. Prior whole-row23918/45745 computes different scales and is only context.
No actual grouped correctness, latency or model-quality result is claimed.

Public entry module: `ops_int4_grouped_activation_store`. Both `quantize_pack_input`
and `quantize_pack_hidden` take `(device, canonical_BF16_input, immutable_LUT,
immutable_shift_table, mutable_packed_I32_output, mutable_group_scales,
mutable_relative_HBM_offsets)`. The last two owners use the identical unpadded
`[P512/256,K/960,P512%256]` mapping (F32/I32 respectively). H owns8192B each, L32768B.
Packed output remains the validated blocked I32 wire. Original quantizers and
production callers are unchanged.

Replay both old/new committed source bindings and resource records:
`python3 experiments/int4-grouped-activation-quant/analyze_metadata_store.py`
This validates old manifests against their Git objects (ignored Cargo.lock
against the original current bytes), so later additive module exports do not
misrepresent old builds. `analyze_resource.py` remains frozen with the original
checkpoint and is directly runnable at490135d8.
