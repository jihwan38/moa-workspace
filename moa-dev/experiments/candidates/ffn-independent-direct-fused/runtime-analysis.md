# E135 runtime failure scope

Whole job110314 times out at the first candidate FFN launch (code4); E134 before/after pass30 comparisons with FFN medians314468/314588. Candidate QKV/O pass12 comparisons. Projection job110316 times out at case0 before either up or gate output. No E135 FFN/projection arithmetic comparison or valid FFN timing exists. Both actual uploaded archives match the verified packages.

The production-prefix projection wrapper calls the unchanged input quantizer and new independent direct up/gate, then stores the gathered BF16 rows. It excludes preRMS, GeGLU, weighted down and final epilogue. Their absence in the second failure rules out a uniquely whole-suffix explanation; it does not identify the earliest blocked internal task.

The graph gate verifies compact HBM source/destination coordinates, five disjoint six-row windows per independent projection, aligned scale rows, native→input-scale→block-scale→F32 reduction, 8B output pairs and512×120B F32 gathers. It checks register binding/lifetimes in the schedule namespace, separately from final-graph IDs. It does not establish hidden native/SFR read indices, bank arbitration or every physical alias lifetime.

The actual compact weight ingress is225×256B=57600B per slice. Each six-row view is45×256B=11520B. Joint60-row FP4 and materialized30-row FP8 buffers are450×256B=115200B. All three extents preserve a128B bank phase. The [official memory documentation](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html) maps DM bank selection to bits3–6 and warns about64 consecutive same-bank accesses while DMA waits. It does not impose an even number of256B packets. Therefore odd225 alone neither proves a bank conflict nor supports padding. Linear packed fetch and three adjacent8B output commits do not expose a direct same-bank repetition violation. Aggregate virtual allocation addresses must not be interpreted as physical bank addresses.

Read-only comparison also found native/DMA/table overlap in passing E131 whole graphs. E135 whole first native322(58586→60649) overlaps the other packed DMA294(53213→77825); its projection first native189(43757→45820) overlaps subsequent table preparation. E131v1 first native381 also overlaps table preparation, and E131v2 overlaps down DMA. These are static schedules, not observed stalled intervals; removing overlap is not yet a causally supported repair.

The frozen [SDK excerpt](evidence/sdk-timeout-profile-excerpt.txt) shows `Trace::wait` calls `launch.collect().await?` before `profile.resolve`. A failed launch therefore returns an error before public span resolution. A names-only profile replay or catching the existing unwrap cannot be assumed to reveal partial progress; no such extra allocation was spent. The source path/hash and read command are preserved in [its origin record](evidence/sdk-timeout-profile-origin.json).

E103's broad wrong outputs and later timeout, M06's timeout, and this timeout are distinct observations. None proves a common compiler or hardware defect. No arbitrary padding, tile, context or ordering repair is selected. This prototype is held unresolved; independent direct decode is not declared exhausted.

The read-only independent [liveness comparison](independent-liveness-review.json) binds these whole/probe/E131 schedules and records register/table readiness checks and their limits.
