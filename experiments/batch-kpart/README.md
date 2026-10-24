H960-partitioned P128 query (R2-BK01)
===================================

`ops_batch_kpart::project_query_wide` partitions the contracting H3840 dimension
across four H960 slices. Each slice owns64 output channels instead of16, so
weight bytes and multiply work are unchanged. Each input is replicated across
64 output groups instead of256, reducing activation DM replication by four.
All64 local token rows fit in122880 bytes of DM and are staged into TRF in B8
pieces without another HBM read. F32 partials reduce across four slices before
the original BF16 dot rounding and BF16 channel-scale boundary.

The F32 reduction order differs from the full-H control. The standalone probe
therefore has two gates:524288 exactly representable dyadic outputs checked
against an independent scalar oracle, and524288 seeded varied-exponent outputs
compared against the existing device kernel. The latter reports bit differences,
maximum absolute/relative error, and failures at `0.001+0.01*abs(reference)`.
Passing this kernel tolerance does not establish whole-model correctness.
Matched ABBA samples use five launches and one identical P128 result read.

Kernel and executable compile. Hardware correctness, numerical differences,
timing, and subsequent actual-model validation are pending root. The canonical
serving kernels have not changed.
