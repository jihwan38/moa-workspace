# Exact O projection K-local rings and H64

Control:8e1f133 K-localring O, validated r542/318660 all independent NaN output bits. Sliding warm .648 to .633ms, full .937 to .769ms. Previous direct-HBM H64 d96a3ac lost .643 to .656ms because of doubled input copies.

Change only sliding H120/128 live slices to H64/240 live slices, preserving the now validated per-K ring64 input broadcast. K1024 BF16 partial tree/scale and four input/output ABI remain. Input HBM traffic stays source32rows perK; weight duplication falls and dots spread across more live slices. Added partial-gather/output replication may offset benefit. Full attention projection unchanged.

Gate: two seeds, two orders, candidate outputs separately NaN initialized, all output bits and finite against canonical partial tree; inclusive ABBA against passed K-localrings. Compile estimates alone do not justify adoption. No host adoption before whole Prompt512 equality and measured win.

## Actual stage attribution

The validated H64 projection remains unchanged. Add one opt-in traced launch after completed ABBA timings; all correctness buffers and timing paths remain unchanged. Current whole-model output projection costs about28ms per512-token group across48layers. Inspect actual DMA/contract/partial-tree/store spans before selecting another change; do not infer bottlenecks from FLOPs or static estimates alone.
