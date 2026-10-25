# Actual first32 FFN and LM gate

Arena319293, root run r842-r32-hotstages-submit, succeeded4.67855s. Frozen a006 source3455dd1, ELF c5a6294ad8df9187939118f917ebe28b967d4466115123f6fa2dc01cbad68cb3. Current R64 arithmetic is the control; no serving dispatch is changed by this micro.

All three seeds passed all first32 FFN outputs (368640 BF16 values total), E4 inputs/hidden values, BF16 hidden, both F32 per-token scales, and untouched tail32. Seed97 poisons inactive input rows with NaNs. All25,165,824 live LM BF16 values matched current partitioned-H960 R64 exactly, with tail32 outputs preserved byte-for-byte. This is a current-arithmetic differential gate, not a new independent full-model quality result.

Complete completed-launch ABBA timing, downloads/uploads excluded:

| Stage | Current64 | First32 | Reduction |
|---|---:|---:|---:|
| FFN |408.274–408.321us|281.517–284.619us|30.3–31.0%|
| LM |3378.382–3378.545us|2740.002–2743.555us|18.8–18.9%|

Actual NPU trace, cluster0:

| Unit |64 count/union cycles|32 count/union cycles|
|---|---:|---:|
| Task|1 /789388|1 /550640|
| DMA|37 /658490|29 /488970|
| TU|49 /178333|33 /97310|
| VRF transfer|33 /26124|21 /12859|
| Intercluster|6 /33332|6 /29289|

Unit spans overlap; do not add categories to obtain wall time. Both clusters remain used by output-sharded FFN. Weight tensors and H960/L960 reduction structure are unchanged. The reduced token work removes16 TU calls and8 DMA calls while retaining fixed weight reads. It does not halve latency. Extrapolation from this single supplied-weight layer is roughly5.9–6.1ms per48 FFNs and0.64ms for LM; whole-step matched histories determine the actual outcome.

Build failures a001–a005 are preserved. Their repairs were physical two-cluster mapping, minimum8-byte scalar SRAM packet, unsupported tiled HBM reshape (moved to DM after actual named-axis load), and private mutable GeGLU tile padding32. They were compile failures, not NPU numerical failures.
