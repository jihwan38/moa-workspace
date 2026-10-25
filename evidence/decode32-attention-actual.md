# Actual capacity32 attention gate

Arena319296, root r843-r32-attention-submit, succeeded1.41028s. a006 source3455dd1; ELF18f04d81745587411580128a82e8067f9ba7fe4df38f26b3043dee93258f1a61. All16 low4 policy combinations matched131072 BF16 values each against the current64-query mixed kernel (2,097,152 values); future poison,17/24/32/7/32 live-mask transitions and untouched tail32 passed.

Complete5-launch ABBA: full live groups plus inactive short groups (policyf0)588.06–597.09us→357.80–366.62us; half short live groups(policyf5)526.59–534.99→298.78–305.44; all short(policyff)467.36–475.23→237.76–245.49. The removed work is four unused R8 groups; physical Rc64 storage, K/V encoding and BF16/F32 reduction trees are unchanged.

Actual policyf5 cluster0 trace: Task1034861→586983cycles; DMA87→43 operations/union767699→447759cycles; TU112→56/465421→234650; intercluster25→13/65386→32995; memset12→4/12965→4322; VRF32→16/32567→16176. Unit intervals overlap. This differs from the old rejected R16/R32 prefetch variants: those still computed all64 rows.

Independent host/probe audit by round2_path on ec2d829 found no selector/owner/stale-output issue. The existing prefix768 guard requires live>56, so capacity17..32 does not bypass an eligible768 route. Automatic dispatch remains unchanged; explicit diagnostics compare same R64 arithmetic and live output policy. Wholemodel evidence is still pending.
