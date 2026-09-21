# Commit O quantized components directly into the packed DM tensor

The E26 O quantizer previously wrote hi and lo into separate512-byte shard
buffers, then issued two typed DMA copies into the two-component packed tensor.
This candidate commits each quantization pipeline directly into its destination
component tile. The lo pipeline reads hi from component0 through a typed view.
All512 values of both components are written before the packed tensor reaches
TRF; there is no use of uninitialized padding or new unsafe reshape.

Runtime per-column amax, power-of-two scale, FP8 rounding, subtraction, low-part
factor16, two-lane dot, FP32 partial sum and the entire O epilogue are unchanged.
Thus this candidate changes storage and lifetimes, not the approximation's
mathematical definition. The parent's outlier/cancellation limitation still
applies, and a compiler success does not guarantee equivalent NPU behavior.

Exact O compilation passes at26,686 static cycles versus E26's29,088, a2,402-cycle
reduction. NPU validation is pending in the combined E29 experiment, which also
changes the QKV input normalization/quantization layout independently. No speed
or correctness result is claimed from this static number alone.

`projection.rs`, `ops.rs` and `rmsnorm.rs` freeze the standalone candidate with
E26's original QKV path; E29 combines only the O portion with a newer QKV path.
Source/schedule hashes are recorded in `evidence.json`. The exact build uses
`/tmp/moa-output-direct-packed` and its own target/compiler output. `candidate.patch`
contains only the changes relative to frozen E26.
