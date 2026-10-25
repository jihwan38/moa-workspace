# Exact32 O matched whole-model results

Both jobs use the same frozen a013 source662e309 and ELF
cd152b0684c0f460189690ff00502fb59d8003d292a31a17c2438db611b816c4.
Control is the already accepted hybrid32 path; candidate changes only O.
Both use the same R64 LM association, independent KV histories and request order.

- r897 /319467,27.023s:75,759,616 BF16 logits plus greedy exact.32 paired512-token
  histories;11,811,160,064 reserved KV bytes exact after regroup/reset. Warm32
  steps3/6/7: control66.305/65.900/66.249ms versus64.226/64.289/64.144ms.
  O stage alone6.810–6.835ms→4.942–4.973ms, removing1.85–1.89ms as predicted.
- r899 /319475,22.018s:58,195,968 BF16 logits plus greedy exact.17 paired mixed
  histories at0/511/512/767/768/1023/1024/1151 with48 active-P128 interleaves;
  6,274,678,784 reserved KV bytes exact. Warm eligible17 steps1/2/3/6/7:
  control65.994/65.832/66.147/65.896/66.234ms versus
  candidate64.144/64.189/64.179/64.319/64.371ms.

Total133,955,584 logits and18,085,838,848 reserved KV bytes are exact across both
jobs. The logit totals include additional live33/64 position-zero guard calls on
both APIs, where the existing64 projection remains selected. Cache comparisons
are completed before those reset-only guard calls.

First candidate calls are cold and excluded from speed attribution. Live7/16
are explicitly forced through R64 by these diagnostic APIs, not ordinary B16;
33/64 do not select O32 either. Their small timing differences are not optimization
claims. All model intervals exclude vocabulary reads and full-cache validation;
device greedy times are reported separately. The evidence supports about1.8ms
per eligible17..32 model call, not a standalone HTTP/SLO or scoreboard result.

The export on6cd4878 enables O32 only when ordinary Automatic already chooses
hybrid32: independent canonical singleton count17..32 and staged FFN enabled.
The explicit original hybrid32 control remains available. Pools, cache format,
weights, live commits, shape/alias guards and lower/higher-width routes stay intact.
