# Scoped model gate

B16AttentionOutputDiagnostic is a per-call Dispatch value; Automatic remains
unchanged/default-off. It follows identical Automatic P128/R32/R64 and native
output-layout choices. Only pure independent singleton B16 batches2..16 may
compose, and only canonical storage plus Hsplit O. Legacy/scalar/R32/R64 and
other O layouts retain old paths. No persistent flags/scratch/metadata added.

Clone the actual retained-output/all-cache metadata model oracle:17 paired
request histories in disjoint0..16/32..48 rings, reversed fullpages, activeP128
spare owners50/51,48 layer interleaves, widths1/2/8/14/16/17, reset0, explicit
O/QKV fallback then re-entry, narrow16+15 prompt reuse. Both passes keep current
prefix/epilogue/compact metadata. Read all logits and greedy/stochastic results
before next scratch overwrite, compare all cache bytes afterward. Args512 and
1024 select separate bounded complete-history jobs. No serving activation.
