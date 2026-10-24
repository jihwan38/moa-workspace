Actual NPU job318071, run r302-b16-output-parallel-submit, froze source45f56f2.
Probe binary SHA2560671e5a9597326b552471ebb899ef19454642983be87ba813139c837ae71cd5b.

Sliding and full each passed all61,440 BF16 output bits against original B16
for seeds0,31,79. Every dyadic CPU K1024 partial and final channel passed; both
nonrepeating random cases passed sampled independent CPU partial/tree checks.
No mismatches occurred in zero/cancellation rows or signed/nonunit channel scales.

MatchedABBA warm launches (20 per leg, no output read inside timing):
- Sliding control215.239/214.503us; candidate86.936/85.906us.
- Full control443.432/443.630us; candidate122.788/122.355us.

This proves the isolated kernel arithmetic and a kernel speed gain. It is not
a full-model, public accuracy, latency-SLO or leaderboard result. Root owns
host adoption and subsequent model/evaluation gates. Dependency-free export
d3e3f6f contains only module/probe/lib registration and experiment plan.
