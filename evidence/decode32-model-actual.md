# Whole-model capacity32 gate

The exact control is forcedR64 on both sides, including live7/16 fallback. These are same-precision route timings, not the ordinaryB16 decoder costs at those widths. The existing Rc64 cache and Blocked64 output/greedy ABI remain unchanged.

##32 paired owners

Root r856-r32-model32-submit, Arena319312, succeeded26.4015s. Frozen a007 sourceec2d829 and ELF8fb279a7501841df12ebbc17f47223533dbe54dd023cf5da04682644cccebe06.

Both sides independently computed512-token histories for32 owners (32768 real input tokens). Across32→17→24→32→7→16→32→32 regrouping, order permutation and position0 reuse, all50,331,648 live BF16 logits and device-selected tokens matched. Every reserved K/V byte matched:11,811,160,064 bytes on one side, all40 entire sliding rings and all4 reserved pages of8 full layers per owner. Cache comparison9.753s is excluded from model timing.

| Step/live | Current full64 model ms | Capacity32 model ms |
|---|---:|---:|
|0/32 cold candidate|83.395|72.773|
|1/17|81.804|65.240|
|2/24|80.744|66.199|
|3/32|82.633|66.539|
|4/7 unchanged forcedR64|76.260|77.202|
|5/16 unchanged forcedR64|77.582|77.378|
|6/32|82.323|66.646|
|7/32 reset|82.754|66.663|

Warm32 improves19.0–19.5%, approximately15.7–16.1ms. Selector time remains approximately1.2–1.4ms. Model timing includes all launch/wait/metadata work and excludes full output downloads and cache audit. The first candidate dispatch is cold and is reported separately. This supports the complete-path gain predicted by the independent FFN/LM/attention microgates; it is not an HTTP throughput score.

##17 mixed owners and active prefill

Root r859-r32-model17-mixed-submit, Arena319322, succeeded21.809s with the same a007 ELF. All32,768,000 logits, greedy selections, and6,274,678,784 reserved cache bytes on one side matched across true0/511/512/767/768/1023/1024/1151 histories, reordered17 rows,7/16 fallback, position0 reuse and48 advances of an independent active128-token prompt. Warm eligible17 model calls improve81.39–81.99ms to65.85–66.13ms; first candidate69.023ms is cold. These still compare forcedR64 arithmetic on both sides, including7/16; ordinaryB16 timings are separate.

Total both jobs:83,099,648 live logits and18,085,838,848 reserved KV bytes compared exactly, with no ownership/selector mismatch. These are model/cache gates, not public HTTP quality or throughput results.

##Activation

After both gates, the minimal source-only activation passes the capacity32 option for Dispatch::Automatic as well as the explicit32 diagnostic. Decode64 still requires independent canonical singleton rows; its local eligibility remains17..32 and staged FFN. Counts1..16 retain the ordinary scalar/B16 dispatcher,33..64 keep current64 hot stages. Forced64 and the existing original-attention diagnostic controls remain unchanged. Rc64/Fc256 storage, live-prefix commits, requests/reservations, all output layouts and greedy/fallback policy remain unchanged. No persistent selector and no Worker change.

The c067 root independently enabled B16 epilogue fusion. That affects the ordinary<=16 route, while this exact gate deliberately uses forcedR64 for those transition controls. Root should recheck combined publicC17/width transitions after integration; no claim that these old forcedR64 numbers describe c067AutomaticB16.

