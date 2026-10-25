Independent batch_kernels review of frozen673096ab passed group_index/inverse row mapping,
unscaled dense-code reference and per960 F32 scale ordering, appended partial exports,
candidate-first eight snapshots, common-history feedback, true row permutation and
completed-call ABBA resets. Independent F64 samples have explicit tolerance; they do
not establish CPU-bit parity. Current-E4 weight/stage errors exclude original NVFP4
prepack loss, as labeled. No concrete reference/ownership/timing issue found.

Candidate8df08ec3 prefix and intended dense transform were mechanically verified.
All five new/control device entries compiled in a003; source hashes match. No host
model, Worker or production storage field is changed. Root alone dispatches.
