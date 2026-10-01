# E154 independent compiled gate

`audit.py e154-replicated-v1 --check` passes. The 46 actual compile inputs are bound to E151 plus the frozen E153 three-file composition and the reproduced E154 patch. The original E151 source proof and emitted RMS operations O0–O9 remain intact; ops/RMS bytes match E153. QKV and O images match E151.

The new O12 global Max still uses actual ratio256 over eight valid cluster0 H480 shards and padded positions, then commits the scalar to eight Dummy8 outputs. O13 relabels those same live positions as H480 owners; O14 applies the original exponent mask and multiplication. O15 reads that DM directly into VRF, so the former quantization scalar DMA is absent. The separate restoration DMA remains: O49 reads only initialized replicas0–7 and writes one32B packet to each of512 consumers. All reshapes retain the physical/live slice mask.

Actual late packed replication remains32×720B=23040B; three Sub copies retain480B lengths at0/480/960 within1440B. Original ring seed/broadcast, four native streams and five E151 scale streams preserve their contracts. Every overlapping VRF lifetime has disjoint addresses inside8KiB per slice, and registered inputs survive their consumers.

The independent raw graph review binds the complete production MLP prefix plus the unchanged original up/gate wrapper and host. It checks the same Max8/direct-VRF/restoration and packing contracts in that separately compiled diagnostic. The raw oracle excludes preRMS; whole source/emitted RMS identity and the full FFN test cover a different scope.

These are source/layout/register checks. M06's related replicated-Max attempts had unresolved SFR failures; this exact target compilation succeeds, but compilation does not prove hardware liveness, arithmetic or performance. NPU results and final adoption validation remain separate. No profiled wait is claimed as entirely recoverable time.
