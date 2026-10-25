# Complete compact32 model gates

All11 grouped-owner cases and four history cases passed on frozen07718f07.
Root dispatched/collected r1210..r1224; no source change occurred between jobs.
all-model-results.json preserves each log SHA, submitted ELF, timing, memory and
comparison counts. All131 package tests passed with0 filtered.

Group/Single/Pair/Tail256/Tail384/Three cases compare64,225,280 logit values and
27,179,089,920 reserved-cache bytes, including two order/reset rounds, NaN tail
padding, sentinel isolation and case0 paired B16 interleaves. History cases512/17,
1024/31,1024/32 and8064/2 compare51,380,224 paired logit values and21,063,794,688
reserved-cache bytes. The17-row case retains an independent128-token prompt
checkpoint through six R32 decode calls;32-row case remaps physical rings in a
second round;8064 history keeps all16 full pages and tests reset near8192 context.
Both the model forward guard and cache diagnostic reject logical slot32 when
Workspace requests=32. These are full provided-model computations, with identical
current c079 arithmetic in both paired arms.

This establishes compact32 internal paired-owner parity and cache isolation.
It does not yet establish cross-program equality with the old physical64 model:
the original64 executable is independently building with the same fixed32-owner
schedule, and its ordered SHA256 stream must actually be compared after execution.
No public task/serving throughput or more-than32 queue latency claim is made.

Available warmed memory stays approximately14.276GB after joined FFN weights and
lazy model scratch. The physical saving is exactly10GiB of KV payload; matched
allocator-level64-versus32 availability remains a separate pending measurement.
