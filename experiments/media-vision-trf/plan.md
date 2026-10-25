# Exact B8-TRF vision projection

Basea98ca2b; no host activation. Current passedB16 module1818875/r777 runs one token per TU lane with B%8 distributed across output slice groups. FusedB16 actual3.008M raw Task units includes DMA interval union2.860M (~95%); this motivates weight reuse, not an assumed theoretical gain.

New mapping: Cluster=B/8; Slice=(Mv/120,1#8) or(H/120,1#8); TRF lanesB%8 andK768 (12288B); each weight120×768 BF16=184320B. A normalized input gathers its8 token rows into DM elementB8×K and broadcasts to32 real V32 groups, then zero-copy relabelsV32 to the output row axis. Weight DMA no longer has B%8 in its slice mapping. Each K768 dot casts to BF16 and uses the identical nine/five-part balanced BF16 addtree, bias boundary and normalization.

Full BF16 input resident8×6912×2=110592B/slice. Weight184320B and nine partials9×120×8×2=17280B plus live vectors fit nominal512KiB DM; compiler allocation is a separate gate. Scalar and currentB16 paths remain available. New module and diagnostic probe only; own target, root alone dispatches NPU.

Gates: actual provided visionweights; all-stage/fused BF16 bits, separate normbits, exactF64 samples/standard frozen F32 accumulation interval, rowpermutation and padding, kernel-only and upload/allocation-inclusive16patch ABBA against BOTH scalar and currentB16fused. Trace wholecandidate; no speedclaim from compile/traffic count.
