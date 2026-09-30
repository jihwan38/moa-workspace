# Constant-bool tensor-return branch evidence

The frozen v1 [source](qk-half-v1/projection.rs) calls `project_native_kv` with `true` for K and `false` for V. Both are tensor-valued `if staged` expressions. Its unchanged-source diagnostic [vISA](evidence/e146-qk-half-v1/failure.visa) still contains both constant branches and two tensor phi nodes:

- K: `PhiFunction { lhs:399, rhs1:395, rhs2:398 }`; staged WriteView versus full native Fresh output.
- V: `PhiFunction { lhs:429, rhs1:425, rhs2:428 }`; the corresponding false-branch selection.

Both phi results feed the canonical-head `LowLevelReshape`. The [machine-readable audit](branch-phi-audit.json) binds the exact source, vISA and original failure log and records byte offsets and small exact excerpts. Reproduce with `python3 branch-phi-audit.py` (no compiler or NPU calls).

The failed scheduler names `T170 := Scratchpad`, `wait_targets:[T185]`, `size:0` do not occur in the earlier vISA ID space. Their direct identity with either phi is **unproved**. The declared Q/K raw F32 shapes have nonzero live sizes of 16 KiB and 8 KiB, so the allocation request is not proof that the logical output is empty.

Separating unconditional staged K and original V functions removes these observed phi nodes without new copies, fill or numerical changes. This is the specific source-expression hypothesis tested by v4. The two earlier initialization/compact-copy repair failures remain separate evidence; no target-wide staging limitation or hidden allocator cause is claimed.
