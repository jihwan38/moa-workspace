# Exact output activation ingress experiments

The original O32 activation DMA replicates an input tile across output-channel shards. r504 attempted a few-source HBM load followed by full DM broadcast but exceeded SRAM. Sourcefe67220 tiles ingress to32rows so source and destination each consume64KiB per slice and compiles (r509).

Arena318556/r513 passes every independent NaN-buffer BF16 output comparison (two seeds, two request permutations, sliding/full). It is slower: sliding0.939ms versus warmed0.646ms O32; full1.416–1.419ms versus0.943ms. The intermediate DM broadcast is rejected for production. Logical HBM traffic reduction alone is not an end-to-end gain.

The bounded follow-up distributes input queries/Kshards across source slices (2KiB each), then uses Switch in the sub context directly into the unchanged32-row TRF, avoiding the large destination DM copy. All MAC, BF16 partial gathering, addition tree and scaling remain unchanged. The same strict bit gate and inclusive ABBA decide adoption.

The direct Switch candidate991d8c7 also passes all independent-buffer BF16 comparisons (Arena318587/r527), but is slower: sliding0.944–0.948ms versus warmed0.648ms; full1.111–1.113ms versus0.943ms. It is not adopted. A compile success and lower logical HBM replication did not establish a whole-kernel gain in either O ingress variant.
