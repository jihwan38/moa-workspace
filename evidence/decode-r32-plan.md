# Exact 32-row decode feasibility and bounded experiment

Source baseline: 6f39574, functional a052254. Storage remains Rc64/Fc256; this is query work reduction, not cache capacity reduction.

Actual serving evidence: runs/r805-c064-gsm-submit/arena.log line76 reports425 width17 decode calls mean75.22ms/max79.60, versus width16 mean46.75ms. Source f8fc9d3 and frozen binary0d9df13d. c060/c062/c063 show the same cliff. These are autoregressive scheduler call timings, not isolated model kernels. The older r583 source0679b6a real64x512 history profile reports85.46ms model, sliding attention+commit29.815ms, FFN24.162ms, LM3.387ms. It predates current exact optimizations and cannot predict current17-row costs.

Verified current control: Model chooses Decode64 for independent canonical singleton counts17..64. Step64 masks padding but all64 dense rows execute. Mixed512 attention executes all8 R8 groups; unused groups still run prefix512 attention. Partial commits extract only live B16 chunks. R64 FFN already uses output-cluster Up=L/7680 and Down=H/1920; no new output sharding is proposed. LM uses H960 partial F32 reduction and differs slightly from B16 LM, so two B16 calls are not an exact R64 replacement.

Prior experiments r223/r231 changed64-request attention prefetch width to32/16, passed bits but slowed714us to754/762us. They did not implement32-query execution. No prior true R32 FFN/dense implementation found; batch-kernel owner confirmed.

Bounded route: retain Decode64 scratch, existing output/greedy APIs and all64 physical slots. Add capacity32 hot-stage kernels: four existing R8 attention groups with unchanged causal/prefix arithmetic; exact R32 FFN token-axis work with the same H960/L960 reduction, scales and HBM stage boundaries; first32-row LM tile with unchanged H960 association. Norm/QKV/heads/O/full-attention initially remain64-row kernels. This is explicitly a hybrid32 execution policy. No new precision, request reuse, weight copies or cache owners. FFN and LM weight reads remain; their work cannot be assumed to halve wall time.

Expected removed work: four inactive R8 attention groups, half FFN token MACs/activation and intermediate stores, one LM32-row contraction/output block. Current all-short attention471us/64 implies at most roughly9ms/40layers of removable half-query work before fixed costs; this is only a mechanism estimate. Actual whole-step measurement decides adoption.

Gates: candidate-first output poison, all first32 FFN/LM bits against current R64, untouched tail32, all16 low4 attention policies, finite/future poison and cache untouched. Then same-schedule32-capacity versus forced64 wholemodel matched histories, all vocabulary logits and complete KV, widths17/24/32 with7/16/33/64 regroup, ring wrap, reversed pages and active prefill interleave. Host flag is per-call and default off until proof. Root owns all NPU dispatch.
