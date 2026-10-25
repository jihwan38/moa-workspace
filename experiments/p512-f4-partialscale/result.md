# Complete composed resource gate: PASS, no P512 hardware claim

Source `dffea370`, base c079 `394d60a2`. Both exact resource compilations used the
same tree and target; all source hashes remained unchanged. Candidate compiled
in21.63s, control21.36s. Commands, hashes and logs are retained in the a001/a002
manifests; schedules are the actual compiler outputs.

| Compiler quantity | Current E4 | F4 group960 |
|---|---:|---:|
| Complete scheduled finish |1,076,934|917,695|
| TU operations / summed cycles |128 /748,727|140 /720,467|
| HBM→DM operations / summed cycles |51 /415,232|65 /330,486|
| DM→HBM summed cycles |84,353|84,353|
| DM→DM summed cycles |27,023|27,023|
| Lookup-table loads / cycles |0 /0|12 /15,516|
| DramStack allocations |0|0|

Complete estimated finish improves14.7863%, clearing the fixed10% resource
gate. Source-identified joined-UP/GATE and DOWN weight loads each decrease from
97,765 to49,157 estimated cycles. The candidate includes new group-scale loads,
fixed-code lookup, local partial multiplication and reduced-F32 materialization.
TU counts increase because the original row/xscale BF16 epilogue follows that
new materialization. Aggregate operator cycles overlap; their sums are not an
actual latency decomposition or a device throughput claim.

The GeGLU/global scaling/amax/activation-quantization helper region is
byte-identical to the current core (hash in `unchanged-source.json`). Current
norm-store and residual modules and the original control are untouched. The
projection axis and joined output aliases are imported from the original core,
so downstream scratch geometry remains the same.

This changes static weight quantization and multiplies each local K960 F32 dot
by its group scale before the existing inter-slice sum. The exact intended
reference is dense E4 expansion of the same fixed F4 codes with the same local
scales; it is not the original E4 model. B16 intended-reference proof does not
establish P512 correctness. P512 micro, complete timing, model and task quality
are still required before any serving integration.
