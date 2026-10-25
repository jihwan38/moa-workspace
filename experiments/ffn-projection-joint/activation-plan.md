# Pending activation patch — not applied

The attached patch is contingent on all11 current-stationary-O/FFN joint hardware gates. It does not change the frozen source or executable.

`Worker::new_with_storage` already prepares grouped scratch before serving canonical requests. Immediately afterward it will prepare the48 joined immutable UP/GATE weights once; failure propagates out of startup before any requests. No scheduling, queue, credit, admission, page reservation or request loop changes are proposed.

The three real prepared-state constructors (`prepare_rows`, `prepare_tail256`, `prepare_tail384`) select projection-owned FFN only when complete joined storage is present. The Prompt512, pair512 and three-owner512 wrappers inherit `prepare_rows`. Models that never prepare storage retain the prior implementation without allocating5.7GB lazily during a request. `GroupExecution::production` remains a neutral false value overwritten by the actual tail state. Explicit per-state false/true controls remain available before layer0.

Disabling composed FFN or global preload clears the incompatible projection selector in all three prepared-state types. Seven legacy PreparedGroup fallback methods clear it when changing any blocked-residual/current-core prerequisite. Prompt512 and pair512 wrappers delegate to those fallback methods instead of mutating their inner flags directly. This preserves old diagnostic execution; re-enabling composition alone does not silently re-enable projection ownership. Attention/history/O-only controls retain the selected FFN because they do not violate its contract.

The existing `advance_rows` pre-mutation guard remains unchanged and independently checks storage/layer count and every execution prerequisite. Publication of joined storage remains atomic after all48 device joins and output allocation. Both stationary-O defaults and producer branches remain unchanged. After the joint gates pass, apply separately, inspect the complete diff, rerun all111 library tests and have an independent activation review before root builds the serving candidate.
