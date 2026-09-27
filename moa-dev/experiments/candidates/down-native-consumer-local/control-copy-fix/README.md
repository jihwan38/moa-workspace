# E140 control-copy-fix

Source/control E137334a705, E140 branch start e35ea35. Only prepare_ffn_fp8_split's three padded-destination term copies change. Exact public Sub480B copy replaces emitted1440B source reads; all scalar math, layouts, weights, native contractions, E137 down and public interfaces remain unchanged.

This is an isolated source-contract correction within E140 diagnosis, not a new architecture series. Whole v4 failed while its actual down9 oracle passed; the comparison will test whether fixing this legacy prefix defect alone changes whole behavior. It is not yet a sole-cause assertion. Full graph, byte bounds and stopping conditions were recorded in recipe before build.

Actual c09 target is isolated from agent's m03 local-v5 build. No runtime/baseline adoption is performed.

Exact compile PASS (151051 static cycles); host contains the exact candidate FFN and unchanged E137 QKV/O images. The independent emitted gate confirms O30/O35/O40 each read/write480B, 15×32B packets, packed offsets0/480/960, and1440B exact coverage. Literal source reconstruction proves that these three copies are the only E137 changes.

Arena111461 passed all45 standard comparisons: FFN315184→314397→314934, bracket315059, **−0.210119%**. This single allocation is not evidence of a resolved0.21% performance effect. The concrete result is a source-bounds-corrected E137 that passes the original three whole inputs. Root owns validation/adoption; no live source or baseline changed here.

The local-v5 combination separately passed45 while regressing3.098140% in job111459. It shares this prefix correction but also changes down ownership/preparation. Those two jobs are not a direct paired candidate comparison. Rawdown111393's9cases excluded this upstream defect and final FFN down-global-scale application; a passing standalone helper cannot override whole prefix memory analysis.

Recheck: `python3 moa-dev/experiments/candidates/down-native-consumer-local/verify_control_copy.py --check` and `python3 moa-dev/experiments/candidates/down-native-consumer-local/control_results.py e140-control-copy-screen 111461 --check`. `control-copy-decision.json` preserves the finite scope and pending root adoption review.
