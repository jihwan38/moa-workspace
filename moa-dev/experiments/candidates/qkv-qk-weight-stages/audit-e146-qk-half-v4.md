# E146 v4 independent compiled gate

`audit.py e146-qk-half-v4 --check` reproduces the frozen PASS in [audit-e146-qk-half-v4.json](audit-e146-qk-half-v4.json). It binds the exact build, 46-file source manifest, graph, schedule, image and E144 reference. Only `projection.rs` changes; quantization, four-term precision, restored channel scale/first BF16, head/cache and O suffix remain unchanged.

Actual five HBM ingress descriptors cover Q 15 MiB and K/V 7.5 MiB each exactly once. All encoded packets are 256B. Indexed source offsets are included: Q stage offsets 0/61440B; K 0/30720B. Each destination slice holds complete H3840 rows under low-two-bit row ownership. All32,768 row-quarter packets form a bijection into the existing K4 native consumers; every 960-element packet stays affine and in bounds.

Q stages write disjoint F32 byte ranges [0,64) and [64,128) per live slice; K writes [0,32) and [32,64). Native commits remain 8B pairs after coefficient multiply → Term reduction → K4 sum. The q0 valid mask, full raw allocation join and one canonical gather per projection match E144. No memset, Sub assembly or whole-weight transpose was added. All inherited non-weight/non-QK-native operator signatures are preserved.

The five native consumers share the same scheduled TRF/VRF operands, whose allocation lifetimes contain every consumer. Mutable destination aliases absent from the schedule allocation table are checked through graph IndexAccess/IndexWrite ancestry and bounds, not treated as independent register allocations. Actual stage/native ordering may differ from E144; neither static overlap nor the lack of extra copies proves a gain. Hidden hardware read-index/bank behavior and arithmetic correctness remain NPU gates.

The direct mutable commits compile after unconditional K/V helper separation. [branch-phi-audit.md](branch-phi-audit.md) preserves the earlier constant-bool tensor-phi evidence and the limit on attributing the private scheduler failure.
