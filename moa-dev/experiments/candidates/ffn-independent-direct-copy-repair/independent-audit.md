# Independent exact E141 contract review

`independent-audit.py e141-direct-six-row-copy-v1 --check` reproduces the frozen source, graph, exact-image and schedule checks. The helper reuses the original E135 compact-ingress/native/gather/register checks and adds the missing up-quantizer transfer/control-bitmap contract. It does not grant hardware correctness or performance by static inspection.

The source differs from original E135 only in the three up-quantizer copies. Each emitted Sub stream reads and writes480E5 bytes as15×32B packets. The indexed destination ranges are0..480,480..960,960..1440, exactly covering the packed1440B allocation. Whole packed base remains17664B/slice, so its last write ends19104. The activation Switch bitmap remains19200..19232B/slice and is now outside every copy. Its actual embedded ring32 one-hot payload and load/consumer dependency are checked. Keeping the same nearby addresses makes this a direct test of the previously identified bounds hazard.

All original up/gate matrices retain exact public HBM coverage and canonical cluster/slice row ownership: each FP4 source29,491,200B and each scale source3,686,400B. Ten fused native streams consume six aligned rows per tile, covering offsets0,6,12,18,24 for both projections. Their actual flow remains directFP4 lookup→three-term native→F32 input scale→F32 block scale→H reduction→8B F32 row pairs→canonical gather→BF16. The gate checks TRF and both VRF bindings, ranges and native-consumer lifetimes; it does not infer hidden bank arbitration or SFR behavior.

A draft checker initially counted unchanged down copies together with up copies and wrote a provisional failed record. Restricting selection to actual H3840/H480 input ownership corrected the checker; kernel/source/image did not change. The package interruption and host reuse are recorded separately. The final successful audit is immutable.

Whole correctness and the original up/gate oracle must run on these exact images. If they recover, that is evidence for the copy corruption explanation in the measured scope; no broad SDK-fault or speedup claim follows from this gate.
