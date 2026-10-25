Basece88461 includes independentfour,Prompt512 and two-owner384+128 constructors.
Kernelchain keeps all dense/attention/cache ABIs unchanged. Added blocked-state
consumers: same H480 QKV norm then original row-wise quantizer; sliding H64 O and
full canonical O postattention+residual+preFFNquant; provenblockedpostFFN. Pack
once beforefirstlayer and unpackonce afterlastlayer restorescanonicalownerfinish.
Micro must compare complete QKV bytes/scales, postattention FFN bytes/scales and
BF16 persistent residual after both layer types, including consecutive feedback.
Arena profiling must use submit --env FURIOSA_OPT_PROFILE=trace (r711 omission).
No host adoption until the consumer-chain gate passes.

Host plan: per-PreparedGroup blocked_residual flag, effective only with default
fusednorm/blockedoutput/shardquant/quarterpre/post choices; old original_* modes
stay canonical for the entire execution. Their future performance comparisons
therefore include storage fallback, explicitly not isolated old norm timings.
original_blocked_residual differs only storage and is the primary exact control.
FFNcore selection remains independent. Active group owns wholeblockedstate across
yields; independentdecode uses separatebuffers. BothKVowner guards and final
row2/3/quarter3/fouroutputs remainunchanged. Cancellation/error retains discard
contract. Nextprepare fullyoverwrites all512values beforea blockedread.
