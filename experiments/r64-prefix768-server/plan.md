Source base:30038f5 with isolated prefix768 kernel a0b4b5c, hardware gate318791.
Only R64 canonical BF16 sliding attention dispatch changes. Step64 owns a boolean
that permits the prefix768 kernel when all live positions are below768 and no
live R8 group qualifies for the existing prefix512 path. This preserves both
all-short and mixed512 priority, including padded/inactive groups. Any position
>=768 or wrapped history keeps its original route. Commit order, masks, ring
indices, request independence and full attention are unchanged.

An explicit original-sliding diagnostic disables only this boolean in local
per-call metadata; there is no persistent Model mode. Both controls force R64,
so current LM/projection arithmetic matches even at widths below17.

Whole-model probe runs two sequential passes through the same Model/workspace,
recomputing every prompt independently and using different physical slot/page
permutations. It stores all control BF16 logit bytes on the host. Uniform mode
fills64 real512-token histories, then64→64(reversed)→7→64(permuted). Mixed mode
covers0/511/512/639/767/768/1023/1024 histories and selection/regrouping across both
thresholds and ring wrap. Inputs are deterministic teacher-forced tokens per
request/position. No paired B16/R64 arithmetic comparison or numerical tolerance.
