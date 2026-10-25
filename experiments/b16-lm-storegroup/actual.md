# Actual exact-store gate: PASS

Root run `r1135-b16-lm-storegroup-submit`, job320744; frozen source9e3d22a8/ELFf060796f. All25165824 supplied-embedding BF16 logits are bit-exact across three non-dyadic original-final-RMS fixtures and two row permutations. All384 selections match for empty/default/arbitrary/all-banned sets. All6144 independent F64 sample comparisons pass; normalized inputs remain immutable. Candidate-first separate NaN destinations and every vocabulary address are checked. This is the exact store-only candidate; the separate H960 association experiment remains failed/held and is not part of the export.

Twelve complete-kernel means: original3380.930–3408.393us; exact grouped-store2374.184–2409.582us. Medians3393.494→2385.768us (~29.7% reduction). Includes the real new DM movement and final canonical HBM store. No HTTP/whole-model performance claim.

Actual traced tid0/1:

| Component (device cycles) | Original | Exact store grouping |
|---|---:|---:|
| Task |6829517 /6828075|4713140 /4712883|
| Final original/grouped canonical store |2071724 /2087824|13284 /13190|
| New immediately preceding DM grouping |none|46006 /46004|
| TU count |97 /97|97 /97|
| TU span sum |564735 /564625|564757 /564613|

The tail's actual~2.07M-cycle fragmented store is replaced by~46k-cycle real DM grouping plus~13k-cycle store. This is the measured mechanism. Input-DMA duration varies between launches despite identical source, so whole Task differences must not all be attributed to the tail. Runtime spans lack source IDs; grouping/store attribution is the ordered final two DMA operations in the compiled graph after the original final transpose. Both kernels retain identical97 TU operations and all math boundaries.

Resource compile reports no DramStack or new HBM scratch; max assigned SRAM end201326592B across512 PEs (~384KiB/PE), within512KiB. Candidate standalone entry remains `[B,H]`, originalBF16embedding `[W,H]`, writablecanonical `[B,W]`; no persistent state/cache changes. Independent batch_kernels source audit confirms the actual named-axis W32→W128 movement preserves every vocabulary row without aliasing. Clean integration changes only the existing B16 decode LM launch; scalar, R32/R64 and prompt-final LM routes retain their existing implementations.
