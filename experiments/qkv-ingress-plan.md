# Exact shared sliding QKV activation ingress

Control: canonical blocked producer from6d58608, already whole-model/public validated. Four groups retain weights and share one input TRF across Q/K/V. New source replaces only duplicated HBM activation ingress with P64×H4 source slices (960bytes each), Switch256 into identical TRF. This is the measured exact FFN ingress mechanism applied to actual sharedQKV; not the rejected blocked-HBM or whole-four-input-prefetch approaches.

Removed:64-fold HBM input duplication perP128, four groups. Added: packetized Switch into TRF; same H960F32partial reduction, dequant/BF16/channel boundaries and blocked outputs. Dot/reduction/store functions unchanged.

Gate: two cases/four distinctgroups/twopermutations, every quant andscale againstCPU, all Q/K/V against canonical independentNaN outputs plus F64 dot checks. Inclusive ABBA includes optionalunchangedquant, allprojectionoutputs completed. No model adoption before measuredwin and wholePrompt512 gate.

Sliding318701/r565 allindependentNaNbits+CPU/F64 PASS; QKVonly .418–.422→.268–.271ms. Printed timing scope mistakenly mentioned removed head calls; source actually times QKVonly, label corrected here. Quantinclusive .514→.365ms. FullQK same ingress now gated separately against canonical blocked output.
