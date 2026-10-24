The isolated output transfer change passed the root's Arena317610 probe.
All65536 query outputs and61440 FFN outputs were bit-identical to the previous
kernels. Matched warm ABBA samples (five launches plus one final read, divided
by five) measured:

* query: original1.23–1.24ms; row-major0.245–0.257ms.
* FFN: original1.95–1.97ms; row-major1.315–1.342ms.

The canonical `project_query_batch` and `batch_ffn` now use the row-major final
DM placement. `*_strided_control` retain the previous source for paired tests.
The same pure layout change is applied to sliding K/V, full Q, and full K.
All five canonical kernels compile. This removes a measured DMA bottleneck;
remaining FFN work includes repeated weight dequantization and other transfers.
The whole-model validation remains root's responsibility.
