# E138 — consumer-local native QKV

Source/control E137 `334a705`; actual start `d6e6c62`. Before-build full graph, coordinates, added/removed logical bytes, register lifetimes and three separate decisions are frozen in `native-v1/recipe.json`.

Each of128 row-owning groups holds four H960 shards of the same actual preRMS BF16 input. Local quantization with common H3840 amax creates fixed four E4 terms, held in one shared activation TRF. Original row-major E4 Q/K/V weights stream natively; weighted packet Term reduction then K4 F32 reduction precedes canonical head gathering, common/channel scale restoration and first BF16. Head RMS/RoPE/cache and entire O/FFN sources remain unchanged. Q/K/V reuse the same prepared terms and TRF by using the exact same Cluster/Slice types; actual emitted lifetimes must confirm it.

Unlike rejected E136, weight TRF staging is absent. The actual E137 control already had no weight TRF: savings against E137 concern full-H BF16 replication and lookup/routing, with real quantization, K4 sums, F32 gather and separate scale passes added. The K960 source stride and output gathers may erase this benefit. No numerical acceptance or performance evidence is inherited from E137 O or E136 QKV. Native-v1 proceeds to whole exact compile and a bounded actual-coordinate/register gate, then early complete QKV NPU comparison.

`boundary-suite/` is owned by root. No live kernel, baseline or official submission is changed here.

First exact compile passed (62,829 static cycles). Independent graph review verified 1024 input960B packets, matching H960 ownership, original30MiB weight coverage, all8192 canonical F32 output rows, Term→K4 reduction and three-consumer shared register lifetimes. Standard host contains exact candidate QKV and unchanged E137 O/FFN images.

Whole Arena111168 passed all45 comparisons but regressed: QKV104,068→117,376→103,860, **+12.900619%** versus bracket103,964. `decision-native-v1.json` separates adoption rejection of this measured prototype, stopped parameter sweeps and a conditional readiness-diagnosis/recovery question. Static12,328-cycle consumer-input DMA and fragmented weight ingress are hypotheses, not observed hardware stall causes. The actual input and head-boundary suite is maintained independently by root; no extra6/performance repeats are requested for this losing frozen source.
