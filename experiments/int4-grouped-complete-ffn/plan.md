# Grouped A4 complete FFN source preregistration

Base is frozen grouped projection `71cdf450`. Import only the exact standalone
`ops_int4_grouped_activation_store.rs` from `f6ad9c83`; record its committed byte
hash and all reused kernel/fixture/control hashes before integration. Neither
donor tree changes. Original complete FFN323067 and drift323102 probes remain
unchanged. Keep the projection donor's `int4_grouped_activation_fixture.rs`;
the complete-chain helper uses the distinct `int4_grouped_complete_fixture.rs`.

The candidate keeps seven completed awaits: original post-attention residual/
pre-FFN BF16 prefix; grouped rational H3840 quant+pack+offsets+scales; checked raw
grouped UP/GATE; original-global/BF16 GeGLU to canonical BF16; grouped rational
L15360 quant+pack+offsets+scales; checked raw grouped DOWN; original post-FFN
norm/residual/layer scalar. Explicit H/L offset owners are8192/32768 bytes, as
are the respective grouped F32 scales. Canonical BF16 stores and rereads remain
charged. No E4 activation intermediate is used before rational encoding.

For each finite BF16 token/group960, apply FTZ, A=max(2^-112,maxabs), exact
q=sign(x)*roundaway(7abs(x)/A), and s=RN32(A*RN32(1/7)). The proved p20 lookup is
only the device implementation; independent CPU u128 quotient/remainder sees
each original BF16 group. Both local projection scales apply before the original
F32 sum: RN32(RN32(F32(integer_dot)*weight_group_scale)*activation_group_scale).
UP/GATE remain pre-global. DOWN preserves BF16 -> global reciprocal multiply ->
BF16. No native E4 row-dequant is multiplied into provided BF16 weight encoding.

All provided layer0 weights, norm parameters, globals, synthetic projected/
residual values and two true row orders match the old complete probe. Two
feedback steps compare every retained output against original P128 prefix,
independent per960 CPU codes/packed bytes/scales/relative offsets, the new P64
dense-E4 intended-order projection, original P128 GeGLU and original P128 tail.
Ordinary typed I4 calls independently check the raw ABI on the same values.
All512 group scale coordinates and packed nibbles are checked; separately
poisoned owners, all input/table/shifts/weights, permutation and full post-reuse
checks remain. Expected initial comparison counts are196,608,000 BF16 events,
6144 i64/F64 samples,40,960 scale values and40,960 relative offsets across the
four correctness steps. Packing covers19,660,800 bytes. F64 envelopes retain
two-product roundoff, first-product FTZ amplification by activation scales,
later FTZ and independent F64 rounding allowance.

The performance control is the unchanged q020
`ops_group_ffn_projection_storegroup::sliding_projection_owned` whole graph,
not the intended-code E4 oracle. Retain complete ABBA with8 restored-input
samples/block and both row orders. Each candidate sample includes all seven
awaits, checked raw validation and every output store. Reset/upload/download,
startup encoding and oracle work stay outside both timers. The existing >=10%
mean whole-chain gain gate must pass in both row orders; old323067 performance
HOLD remains recorded and the threshold is not relaxed.

Additionally record four restored-input phase-clock repeats per row order:
seven completed-await intervals, enclosing total and their residual gap; retain
a separately timed current one-call control. No intermediate validation or
downloads occur inside these clocks. These are wall-clock phase diagnostics,
not a device-per-op trace or a separate promotion gate. Revalidate every output
after all timing/profile reuse.

Report A4 zero fractions and scalar reconstruction error on actual prefix and
hidden BF16 separately from current-production matched-step output L2. Step0
starts from the same inputs; step1 has independently evolved residuals and is
explicitly feedback drift. Synthetic activations plus provided layer0 weights
are not full-model task quality. No layer48 startup/capacity/serving inference.
Only source+plain hostcheck is authorized now; resource/ELF/Arena waits for
component device results and explicit root authorization.
