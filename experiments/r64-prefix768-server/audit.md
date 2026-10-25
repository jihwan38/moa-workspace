Independent read-only review by round2_path passed. Step64 excludes positions>=768,
retains every live all-<512 R8 group on the existing mixed/512 route, and ignores
inactive groups. Commit stays before attention; the original control disables
only the local per-call flag while forcing the same R64 arithmetic. Physical ring
permutations13/17 are bijective modulo64 and three-page ranges are disjoint. Both
passes recompute each full prefix from0. Mixed steps1/2 enter768 after511 crosses
512;767 crosses768 before step3 and correctly falls back. No selector persistence,
stale-pool, control-precision or threshold defect was found. Actual model parity
and completed runtime remain required.
