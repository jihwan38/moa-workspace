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

Actual initial gates:318828 uniform PASS52,166,656logits in27.95s. Whole mixed
318829 hit70s timeout because each independent refill took34.14s; there was no
reported numerical mismatch. New complete jobs split into mixed-low32,
mixed-high32 and mixed-edge64, each retaining separate full control/candidate
histories and all-logit comparisons. Edge uses61→64→60→7→64 and crosses767→768.

The first uniform gate exposed a performance issue at partial width7: old mixed512
used its seven inactive groups cheaply, whereas global768 increased attention
24.09→28.86ms. Eligibility now also requires all8 R8 groups occupied (live>=57),
retaining existing512 priority for every group, including inactive ones. The
uniform64 warm gain remains92.03→90.32ms forward,29.82→28.19ms attention in that
same-run diagnostic. This is not an HTTP throughput measurement. The original
source and ELFs are preserved; the tightened guard and split probes require gates.
