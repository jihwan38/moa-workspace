# QKV explicit prefetch experiment

Starting from the two-cluster QKV and unrolled FFN snapshot, this candidate loads all Q/K/V weights and scales in the QKV entry-point body before calling projection helpers. The helpers receive DM tensors and keep their original lookup, contraction, cast, scaling and gather pipelines. Only the two Stage1 callers use these private helpers. All device entry-point signatures stay unchanged.

The hypothesis was that output-gather DMA dependencies could delay later projection weight loads. Exact isolated compile succeeded, but makespan remained **61994 cycles**, identical to the original two-cluster schedule. No NPU job was run and the candidate was not adopted: changing private interfaces without static gain does not justify an additional hardware experiment. This result does not by itself prove which compiler reordering caused equivalence.

Source patches and hashes are preserved here. The raw schedule is in ignored artifacts/qkv-prefetch/qkv.json.
