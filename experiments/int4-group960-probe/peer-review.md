# Independent source review

Reviewer: `/root/batch_kernels`; read-only, no build or run.
Reviewed d23e82c1 plus the narrow f51e0c1f E4 host-code repair.

PASS: both donor stage bodies byte-identical to pinned7dac9d0c; candidate body
hash unchanged. Per960 weight, activation and output maps align. E4 integer
codes0..7 are exact, all reference row scales are1, group scales precede the
matching4/16 reduction, and DOWN global follows the firstBF16 boundary.
All512-row sign tags/scales and the true permutation cover physical halves.
The i64 local dot bound47040 and conservative F64 interval hold for the normal
provided fixture. Candidate-first separate outputs, ABBA scope and complete
post-reuse host/device immutability checks have no identified confound.

No required correction. The reference retains its disclosed identity-multiply
overhead. Projection timing excludes activation quantization/packing/GeGLU.
DOWN245684>240000 remains HOLD. Source approval is not device correctness,
complete-path performance, or model-quality evidence.

The subsequent7b9aef54 addition invokes the same15-code host guard explicitly
and records an independent exact-Fraction decode; it does not change device
bodies, fixtures, scalar values, reduction order, or thresholds. Host check
passed11.16s with402 unchanged hashed inputs and frozen original Cargo.lock.
