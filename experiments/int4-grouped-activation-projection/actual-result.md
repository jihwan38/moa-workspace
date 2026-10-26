# Actual grouped projection gate323172

Job323172 SUCCEEDED in9.306930598s, exit0. Frozen source71cdf450,
build449ab26f, submitted evidence13867a07. All418 captured source/config
hashes match; original dependencies and frozen ELF4fb53b95…085cd verified.
Both candidate embedded images equal their resource images; both P64 dense-E4
reference images match the fresh build. Root dispatched; agent did not.

Strict intended-code results:35389440 BF16 comparisons per typed/reference
and raw/typed arm;3072 independent i64/F64 samples;17694720 complete row
permutation comparisons per typed and raw arm;56 prelaunch invalid-owner
rejections;571719688 exact input/weight/scale/packed-owner bytes after reuse.
All2048 H and8192 L token-group scales are independently constructed and
checked per row order. The previous provided-weight encodings/fingerprints
remain identical. Counts overlap across arms and are not unique outputs.

| Timed pair | Projection | True mean µs | False mean µs |
| --- | --- | ---: | ---: |
| checked_raw_I4 / typed_I4 | down | 245.507344 | 244.658031 |
| checked_raw_I4 / typed_I4 | upgate | 343.748688 | 343.249312 |
| typed_I4 / dense_E4_intended_reference | down | 245.003594 | 349.530406 |
| typed_I4 / dense_E4_intended_reference | upgate | 344.596219 | 502.105375 |

Each arm has32 completed awaits/projection, in four-block ABBA over two
input orders. Dense E4 is an independent integer-code intended-arithmetic
reference, **not production E4**. Quantization, packing, GeGLU, upload/download
and allocations are excluded. The raw-versus-typed comparison includes valid
raw validation and awaited Function::run, but does not isolate launch overhead.
There is no per-op device trace and no controlled cross-job performance delta.

CPU supplied-weight reconstruction/quantization/audit: 6429.268139ms;
weight upload: 201.322747ms. Declared weight/reference payload
is266,158,084B, not allocator peak or model startup cost.
Fixture creation/encoding/upload records and all16 timing blocks are retained
in actual-result.json and raw log. Wide synthetic token-group scales explain
large absolute DOWN F64 interval radii; strict all-output reference equality
is separate and passed. No original-model drift or task quality was measured.

Decision: intended-code projection and checked raw binding PASS. Historical
complete323067 speed failure and old DOWN resource HOLD remain preserved.
No full-FFN >=10% gain, model promotion, official speed or quality claim.

Replay: `python3 experiments/int4-grouped-activation-projection/analyze_actual.py --verify-local-artifacts`.
All raw-file hashes, counter formulas, ABBA order/scope, ELF/source bindings
and original provided-weight fingerprints are asserted.
