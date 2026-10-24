Canonical workspace startup at control 15a4cf9 allocates and uploads 80 sliding
256MiB pools plus 16 full 128MiB pools with HostTensor::zero: 22GiB of host fill
and PDMA transfer. Public HTTP source c010 reached ready at68.89s, then the70s
Arena deadline interrupted its accuracy request (parent report r241).

This candidate replaces only pool construction with returned HBM allocations.
SDK0.9 DmTensorViewMut::memset explicitly lowers to ParallelMemSet. A4MiB
sliding DM tile or512KiB full tile is filled with +0 and copied to every pool
slot through static tile writes. Each slice holds16KiB or2KiB respectively.
No host view ABI, uninitialized read, altered mask, or request reuse is involved.

Gate: compile both returned output kernels; read every BF16 raw bit in both pool
types and assert positive zero; retain the full80/16 pools simultaneously and
time allocation+initialization, then read the final allocations again. Only
after actual NPU proof may Workspace canonical constructors adopt this path.
Model correctness remains covered separately by canonical/interleaving probes.

Compile evidence: f8020d8 needed a single launch argument rather than a1-tuple;
23c9128 needed mutable `#{!}` padding on the output slot view. 004c966 compiles
both device-returned pool kernels. No constructor adoption before the NPU gate.
