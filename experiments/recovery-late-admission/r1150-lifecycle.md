# Actual c078 late-admission lifecycle gate

Compared completed c077/OFF r1148 (job320874) with c078/late-join r1150 (job320894). The candidate ran52.622s including startup. Both HTTP runs complete96/96 measured requests plus16 warm requests. Every Worker ID1..112 has exactly one completion record. Every HTTP response finishes by length and has exactly its reported64 visible deltas, except the deliberately longer measured request1 with128. All delta offsets are nondecreasing; no missing or duplicate request completion is observed.

`audit_actual.py` checks the collected logs and emits sanitized `r1150-lifecycle.json`, excluding prompt/output text. The frozen c078 source manifest matches all388 files in this tree. The only source differences from frozen c077 are Worker, StartupCohort and the new lifecycle-test file; all model/kernel/Generation/sampling source is identical. Candidate ELF e912924dbfa672d83001a12f9c82e3eff6751c5cff6dd219b67e9d6191d1e423; c077 ELF6ff1b597c7674e3cace7638356bea66beb040bf299f493d2956c72b1d6369333.

## Actual membership and deadline behavior

Seven snapshots contain110 distinct request IDs, with no request captured or joined twice. Five join events append six new owners across two snapshots. Every `previous` list exactly equals the prior membership; current membership is exactly previous plus added, with no removal or slot change. Every appended owner is still Prefill/not-ready with positive remaining work. All existing members' remaining prompt counts are nonincreasing. A held member never completes before its snapshot releases in this length-only run.

The12-owner snapshot65..76 grows through77, then78/79, then80. Each late owner reuses a slot only after the corresponding old passthrough request53,54,52 or64 has a real completion record. The14-owner snapshot81..94 grows through95 and96 after old79 and70 finish. Thus these are actual post-completion reservation/join opportunities, not hypothetical extra capacity. Every snapshot stays at or below16 admitted owners.

For every join, `old_left_ms == left_ms` at the same timestamp: the deadline does not move later. `elapsed_ms + old_left_ms` matches the previously established deadline relative to the unchanged start (within trace rounding), and `oldest_ms + left_ms ==3750ms`. No earlier-receipt shrink or expired-addition case occurred; those were CPU-tested. The first15-owner snapshot with old17 still live has no join opportunity and remains15 owners until release.

The release sequence is ready, ready, deadline, deadline, deadline, deadline, ready (warm included). All-ready releases report every member ready. Deadline releases retain1536,2560,2048 and768 remaining prompt tokens respectively; growing to16 does not mean all16 were ready. Receipt-budget overshoots are7.398,18.814,0.847 and0.230ms. These are elapsed atomic-work overruns, not a renewed deadline. The3750ms policy and150ms gap limit remain scheduling budgets rather than hard wall guarantees.

## Actual old-request progress and visibility

Old52/53/54/64 complete while the65..80 snapshot is active; old70/79 complete while81..96 is active. These ordered completion records establish actual passthrough computation/progress beyond an `allows` eligibility claim. Their slots then appear in added members only after completion. No prior participant is re-held.

The unique(prompt256,output128) tuple maps measured HTTP request1 to c078 Worker17 and control Worker20. Do not equate other Worker IDs with HTTP indices. Worker17 is nonmember passthrough at both ends of the third3037.455ms snapshot. Its first output was already released by the preceding cohort, and it finishes after this snapshot. Its complete real HTTP stream has maximum inter-delta gap174.175ms; therefore it was not hidden for the3.037s barrier. The2910.2ms `hold` on its final Worker record belongs to its initial first-output snapshot, not this later barrier.

The control long stream's maximum gap is157.387ms. Across all96 measured requests, the count with a gap over150ms decreases28→8, but the worst individual gap increases157.387→174.175ms; neither arm has a gap over200ms. The candidate therefore does not establish uninterrupted service at a strict150ms bound. There is no synchronized client/server clock or per-row execution trace here. Visibility versus invisible Decode is not distinguished in the passthrough trace; invisible Decode remains explicitly covered by source and CPU tests, not separately identified in this HTTP run.

## Limits and disposition

The source still rejects held ready members before scheduling (`held && !is_prefilling`), and CPU mixed-completion tests verify no extra hidden decode. Exact visible output/usage counts are consistent with that contract, but do not independently measure every internal token execution. All requests finish by length; actual EOS, cancellation, fatal-error, nonstream and EOF paths are not exercised here and retain their separate127-test evidence.

No lifecycle defect is found in the exercised joins, once participation, deadline preservation, slot reuse, completion ordering or old-request progress. This is a lifecycle gate, not a blanket policy promotion. Matched throughput rises150.377→156.518tok/s (+4.08%) for this stress, while TTFT p90 rises3578.337→3807.943ms. The primary round2_path report owns packing/performance attribution and the separate C16/128 decision. Preserve c076 fixed-recovery regression and c077 control; do not infer all-concurrency or unseen-run stability.
