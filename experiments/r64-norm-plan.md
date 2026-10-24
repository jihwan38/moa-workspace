# R64 normalization and media row insertion

Control: hardware-validated ops_norm_batch, four B16 chunks. Candidate derives ops_norm_wide with Cluster=R64/32, 32 rows × eight H/480 shards per cluster, one row per hidden shard. The same F32 H/480 square sums, divide, inter-slice sum, EPS/sqrt and BF16 boundaries remain. This removes four separate norm launches without changing reductions. Scalar gates retain BF16 residual then gate ordering.

Kernel proof compares all64 rows for RMS, post-attention and post-FFN across two seeds with zero/tiny rows and signed gates, plus independent CPU f64 RMS. Media row insertion checks rows0/31/32/63 and all untouched neighbors. Production adoption awaits NPU proof; this does not change host dispatch.

Initial f284ca0 device build compiled all three normalization kernels, but media insert used H/15 shards (30 bytes) and failed the SRAM eight-byte access rule. The insert repair restores the already-proven scalar row layout used by B16 insert_hidden_row; this helper does not determine batched norm layout.
