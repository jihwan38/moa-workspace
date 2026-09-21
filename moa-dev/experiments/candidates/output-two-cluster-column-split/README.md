# Output projection: two clusters and eight column groups

Exact compile passed: 32,842 static cycles versus 35,557 for the hardware-tested two-cluster full-column projection with parallel residual. This is a 7.64% static reduction. NPU correctness and timing are pending.

Each cluster contains 32 output-row groups × 8 reduction-column groups, using all 256 slices. A slice owns H60 rows × Qs512 columns. HBM weights are viewed as Dummy2 × H1920 × Qs4096 before typed DMA; Dummy2 denotes the contiguous high H axis. Every output row and reduction column is covered once by the Cartesian partition recorded in evidence.json. Each slice directly receives its Qs512 activation segment from HBM, eliminating full-Qs replication and the ring broadcast.

The fused FP8 lookup/BF16 contraction produces an FP32 partial for each row. Eight adjacent slices reduce their disjoint Qs512 partials in FP32 before a single BF16 cast and four-row output packets. Typed DMA compacts H1920 per cluster; only then is Dummy2 renamed H/1920 and the result copied to the existing channel-scale layout. Channel scale, hidden RMSNorm, BF16 boundaries after those operations and parallel residual are unchanged.

The floating-point summation order differs from the previous full-Qs4096 dot, although the real arithmetic expression is the same. This requires NPU and extra-input correctness checks. No test input, weight value or scale is assumed. The original public device entrypoint signature is unchanged; only the private projection helper accepts a canonical HBM view.

The independent compile used /tmp/moa-output-column-split with a dedicated target directory. The raw schedule is ignored under moa-dev/experiments/artifacts/output-two-cluster-column-split. The dominant weight DMA is 13,512 cycles, fused lookup/contraction/reduction stage 5,063. These overlap with other operations and are not additive savings. Source and schedule hashes are recorded in evidence.json.
