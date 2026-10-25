The one full-H E4 mapping compiled without DramStack or internal HBM allocations.
Current exact control estimate2,428,542; native prequantized core1,125,121 plus
established B16 quantizer6,999 =1,132,120 (−53.38%). These are compiler units, not
microseconds or a speed claim. The core has one TRF lifetime across all16 bands,
32→1 epochs and97→51 TU commands. Output storegroup is unchanged.

Startup row quantizer also compiles, estimate3,553,360, no DramStack,16 reads and32
output writes. Its two returned HBM allocations total1,007,681,536 bytes; the
standalone probe reports actual allocator delta and startup time. Historical actual
joined-weight reservation evidence had3,515,547,392 bytes free after complete
media/group/P512/decode warm routes. Subtracting the planned shadow leaves
2,507,865,856 bytes before current-state differences; this is a feasibility bound,
not a new full-model memory proof.

Proceed with exactly one supplied-embedding standalone numerical/performance probe.
Current exact storegroup remains control. No production/candidate-server selection.
The first host check failed because f8e4m3 exposes to_f32(), not to_bits(); the scalar
encoder self-test now compares decoded F32 bit patterns, including signed zero.
The failed check log is retained; no device math changed for that repair.
