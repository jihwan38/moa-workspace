Four-request source-only integration combines two independently gated modules:
retained original QKV weights(318137) and blocked P512 FFN(318138). It keeps
four WidePrefill objects with independent x/norm/projected/raw/head tensors,
PreparedWide masks/positions/blocks and per-request causal cache semantics.

Perlayer execution: normalize and quantize all four request inputs; run grouped
Q/K/V projections directly into the four existing raw buffers; restore each
request's blocks and call the factored unchanged heads/attention/KV/O continuation;
normalize four FFN inputs; join rows; execute the blocked P512 FFN; split output;
apply each original postFFN norm/residual/gate; advance all four cursors together.
Normal WidePrefill still calls the same continuation with its own FFN enabled.

Only complete layers yield. Model::active_allocations guards all four rings and
all pages against other forwards; single or grouped prompt starts cannot overlap.
Model::active_prefill_rings exposes all four. Worker dispatch is unchanged and
never starts a group. Any future Worker integration must retain all reservations
until group finish and use plural active rings. Errors or aborted in-flight
futures can leave partial hidden/KV updates: discard model/workspace rather
than retry a layer. Finish restores each logit workspace swap on Result errors.

compare_prompt_group controls now explicitly use the same native-QKV P128
arithmetic, with identical default prefix arithmetic on independent caches. It
checks all final logits, postgroup C4 decode, active cache alias rejection and
optional same-R64 decode after every layer. Starts896/384/0/128 cover unequal
history, page ends and a sliding-ring wrap; shorter allzero starts reduce gate cost.

The blocked kernel dependencies are the validated source4bd8b07 module
ops_native_fp8_p512_blocks and layouts856ab1c. Four-row hidden pack/unpack
ops_prompt_group_io and test_prompt_group_io were previously source-only held:
these must compile and pass every raw BF16 bit/permutedquarter check before the
group model gate. No device binary/NPU run is claimed for this host integration.
cargo check --bin compare_prompt_group passes. No default precision or serving
scheduler change is included; native QKV public task-quality gates remain root's
separate prerequisite.
