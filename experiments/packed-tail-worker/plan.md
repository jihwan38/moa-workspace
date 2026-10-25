# Full512-first Worker packing with exact384/256 tails

Baseb078362 preserves root c067's independently gated B16 epilogue. Model
sources9b1e19c/180202c ported cleanly as00ac271/8e6acce. Runtime changes after
those ports affect only worker.rs and the optional Tail384 phase timer wrapper
in model.rs. Device kernels, media preparation, four-independent-owner guards,
B16/R64 implementation, admission and startup-cohort modules are unchanged.

Selection retains cyclic oldest-owner order. Existing first-four full512 and
first-two full512 routes precede the new first-three full512 route. Exactly one
of the oldest three aligned owners supplies256, chosen from the earliest
eligible actual owner; the other two supply128. An eligible singleton512 is
preserved before all padded routes. Then three real128 owners, a real256+128
pair, or one advertised384 may use Tail384. Tail256 and ordinary page-local
fallback remain afterward. No owner is skipped past an older short or unaligned
range; no arrival wait, reservation or receipt-budget extension is added.

schedule_inputs restores retained ScheduledInput ranges by ring before choosing
fresh counts. One/two/three actual owners remain reserved through48 complete
layer yields and independent decoder compaction. Route classifiers check exact
owner counts and lengths; the model's real-owner validation remains authoritative.
Output shape and every retained range are checked by unchanged complete_work
before any Generation mutation or cancellation-triggered reservation release.

All real512 and padded384/256 routes use cost class4 for pending credit prediction,
recurring layer feedback and phase tracing. Tail384's measured API splits layer
and finalization timings, keeping preparation/LM/read work out of future layer
costs. Existing emission-based credit, finite startup-cohort membership and3750ms
receipt deadline are unchanged.

Validation:111 library tests pass; cargo check --bin server passes; git diff
--check passes. Seven new Worker tests cover full-route priorities and all three
doubled-owner orientations, every Tail384 shape, malformed classification,
short/unaligned cyclic boundaries, retained ranges after decoder compaction,
atomic cancellation/completion, final-only sampling, conservative cost and the
unchanged cohort deadline/actual-emission credit. Three older selection assertions
were updated because their valid three-row or384 inputs now intentionally pack
more work; the original Tail256 completion test now explicitly supplies an
already retained Tail256 range, proving it cannot expand mid-checkpoint. Initial
failure log is preserved. Independent round2_path source review passed with no
corrections. New model gates are parent-owned; extended HTTP/public evaluation
is still required before making throughput or leaderboard claims.
