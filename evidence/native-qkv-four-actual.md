Actual318137/r334-four-native-qkv-submit, sourcebf742d5.
Binaryaf55ffb181aa1b390938e07a2b3209c89b0f793682c347a41c45e7ae135611a4.
All four input groups have independently validated per-row quantized values
and scales (1,966,080 E4 values perseed). Identity and[2,0,3,1] group orders
match all8,650,752 output bits against current P128 native cores for each of
seeds0,31,79. Independent quantized F64 checks cover all8,650,752 dyadic
outputs and35,840 sampled outputs perrandom seed. Zero differences throughout.

Complete ABBA20 timing includes all four original quantizer launches:
- Sliding fourcurrentnativegroups1998.501/1993.559us → retainedweights
 1553.065/1552.661us (~22.2% reduction).
- Full fourcurrentnativegroups1601.416/1602.762us → retainedweights
 1170.218/1164.344us (~27.1% reduction).

No host row packing or splitting is included or required. This validates the
weight-retention mechanism and preserves the prior native arithmetic; it does
not validate native QKV against task accuracy or a grouped host execution graph.
Root's native-QKV public quality gates and source-only four-request host
ownership integration remain separate prerequisites.
