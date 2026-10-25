# Actual public raw-I4 binding proof

Arena r1192 / job321147 passed in 0.2084 seconds. Frozen source a02a95ff,
ELF a20f8fd7833b91eb5a4793dd534592cf1fdfb964adb90058112e7e3d1cd90bac.
All 4,096 known-byte/raw outputs and 4,096 device-pack/raw outputs match both
ordinary typed I4 and independent I64 dot oracles. All 245,760 device-packed
bytes match, with 983,040 reported immutable input bytes plus separate producer
code/shift checks. Eight fixtures include all signed nibble codes, extremes,
zero, sparse, random, and a true paired row/column permutation. Repeated owners
remain valid; shortening input by eight bytes is rejected by the adapter and
normal runtime (needs30,720/has30,712), with output unchanged.

Typed consumer averaged17.557us (16.583–19.136); raw consumer17.417us
(16.565–20.782, the first raw block includes the outlier); device pack plus raw
58.321us (56.241–60.375). The timed source executes exactly one awaited pack and
one awaited raw consumer per joint iteration, eight iterations per block and
order typed/raw/joint/joint/raw/typed. Uploads, reads and assertions are outside
these timers. This mixed two-kernel sequence is different from r1185's repeated
single pack calls (20.933us mean). Joint minus raw averages40.904us; it is not an
isolated pack measurement. The extra latency beyond separate warm sums remains
unattributed. No second pack exists in the loop; no combined-pair trace exists.

The separate case2 raw-consumer trace records cluster0 Task21,532 cycles:
three DMA durations1,424/2,862/754 and TU ingress/dot7,993/8,818. Cluster1 is the
padded peer (Task17,581); cross-cluster timestamps are not directly comparable.
This trace confirms the normal consumer graph, not producer cost inside a pair.

The public adapter is a validated standalone byte ABI: immutable owning I32
input through await, exclusive distinct output, checked dense geometry/bytes/
alignment, ordinary runtime provenance/signature checks, no typed alias or
unsafe cast. Cancellation lifetime is source-audited, not stress tested.
This does not validate a floating-point A4 quantizer, model accuracy, a model
performance gain, or doubled arithmetic throughput. Production remains FP8.
