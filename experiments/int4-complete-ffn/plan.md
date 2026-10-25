# Preregistered complete FFN diagnostic

Base: bound checkpoint `0ce71c72`, whose actual job323014 passed raw/typed
projection equality. Fused packing comes unchanged from `fe236d83` (actual
323029/323030). BF16 bridges are coordinated with `381fc023` in
`int4-complete-ffn-bridges`; their device behavior still needs this gate.

The real serving caller is `GroupPrefill::advance_rows`,
`composed_ffn_epilogue && blocked_residual && fused_ffn_core &&
projection_owned_ffn && grouped_norm_stores`. Its sliding branch calls
`ops_group_ffn_projection_storegroup::sliding_projection_owned`: post-attention
RMS/BF16 residual/preFFN RMS -> production E4 quantization and joined E4 FFN ->
postFFN RMS/BF16 residual/BF16 layer scalar. That unchanged whole graph is the
same-job performance control. The earlier per960 F4 prototype is not this control.

Candidate, every completed launch inside one timer:

1. Coordinated `prefix_sliding_bf16` (post-attention norm, residual update,
   preFFN norm and actual canonical BF16 write).
2. Validated `quantize_pack_input_aligned` (all512 rows, codes plus scales).
3. Checked public `raw::project_upgate`, with the original owners borrowed.
4. Coordinated `geglu_bf16`, including both original reciprocal-global/BF16
   boundaries and the unchanged Erf/product/DivF2/BF16 chain plus canonical write.
5. Validated `quantize_pack_hidden`, including full L15360 amax and staged store.
6. Checked public `raw::project_down`, retaining its projection BF16 then
   reciprocal-global multiply then BF16 boundaries.
7. Unchanged `ops_norm_blocked_residual::post_feedforward`, including RMS,
   residual and layer scalar. No tail or layout conversion may be excluded.

Only supplied layer0 is loaded. The exact existing `load_mlp`/native prepack
path builds the production control; a visibility-only diagnostic entry exposes
that loader without changing its body. All three provided F4*E4 matrices are
independently reconstructed to BF16 and per960 rational-I4 encoded with the
existing u128 quotient oracle. The old native-E4 row dequant is never applied
to these direct-provided candidate weights. CPU reconstruction/encoding, device
upload and production prepack/join are startup, separately timed and disclosed.
No model weights, responses or calibration examples are embedded.

Correctness uses a separate untimed intended-code path: four original P128
post-attention/preFFN calls on independently canonicalized inputs, CPU exact
rowwise rational codes/scales, independent integer-coded E4 per960 projections,
original P128 GeGLU on the same pre-global BF16 dots, and original P128 tail.
Every candidate BF16 boundary, scale and packed nibble is compared, along with
the updated/final residual. Sampled i64 local dots/F64 rounding envelopes retain
the previously registered projection oracle. The production E4 graph is compared
for finite numerical drift only, never required to equal the approximate A4 path.

Fixtures are one fixed deterministic finite, non-dyadic512-row input/residual
pair plus the true row permutation `(73*p+127)%512`; every special value uses
the source row consistently. Two feedback updates check retained residual state.
Candidate runs first with separate poisoned outputs. Full source/weight/norm/
table/shift immutability and post-reuse snapshots are checked after timing.
All physical layout encodings are bijective, with explicit independent inverses.

Timing: equal initial residual restored outside each completed-chain sample,
ABBA with candidate/control/control/candidate blocks, 8 samples per block in
both row orders. Upload, validation, downloads and optional trace are excluded
equally, while all candidate raw-binding validation/waits remain included.
The preregistered performance gate is at least10% lower complete-chain mean in
each row order. Report all blocks regardless of pass/fail. No model adoption
without a separate full-model/quality/capacity gate. DOWN's failed static screen
245684 >240000 remains HOLD and is not automatically lifted by a runtime result.

Resource and working-set accounting includes original production weights,
candidate I4+scales, independent E4 oracle copies, all scratch and startup peak;
diagnostic duplication is distinguished from eventual production retention.
This phase authorizes source/hostcheck only. Root alone decides a later fresh
device build and one bounded <=70s execution. No speed or quality claim yet.
