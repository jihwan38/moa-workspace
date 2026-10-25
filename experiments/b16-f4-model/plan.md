# Default-off B16 F4 model diagnostic

Base07718f07 is the frozen physical32 branch. QueryR64, fullFc256, context and
current c079 model math remain. The measured B16 kernel8df08ec3 is approximate
per960 static F4 weights with post-local-dot F32 scales and current E4
activations. B16 complete gain23.82% is real; residual L2 error16–20% is serious
and no model-quality inference is justified. No q018 LM or P512 F4 integration.

Model owns an optional complete48-layer shadow, separate from original weights.
prepare_b16_f4 streams one projection: original blockedE4 D2H, exact packed-only
CPU encoder058f785b, then F4/scales H2D. Release CPU source/output temporaries
per projection. Print read/encode/upload separately. Build a local Vec and
publish only after all48 succeed; failure/cancellation drops unpublished owners.
Reject preparation during any active checkpoint, and make completed preparation
idempotent. No dense diagnostic weight output, no original-row/global folding.

Per-call candidate/original modes normalize to Automatic before all unrelated
dispatch/layout guards. Only pure independent canonical singleton2..16 calls
with staged/composed FFN use F4. Pass immutable per-layer references through
decode_batch; never store an execution mode in Prefill. Width1,17+, prompts,
mixed calls and disabled FFN guards remain unchanged. Default source enable
false; explicit candidate rejects missing shadow before any cache mutation.

Bounded model gate: all48 startup/capacity first; original-versuscandidate
teacher-forced2/8/16 drift, finite all-vocab output and own-output greedy,
same-candidate disjoint-history deterministic full logits/cache, activeP128
interleaves, reset/reorder, scalar1/R32-17 unchanged. Strict expected equality
only for identical candidate trajectories and unchanged routes on fresh
matched histories. No cross-original cache parity after approximate steps.
All real owners must fit physical32. Separate source/check/evidence and root
NPU dispatch; no Automatic activation until explicit later authorization.
