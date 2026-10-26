# Authorized diagnostic build

Functional source71cdf450, compiler/check evidencef463ecf1. Root authorized
one fresh standalone test_int4_grouped_activation_projection build after
source review PASS. Use original Cargo.toml/lock, exclusive ffn-k256 target,
CARGO_INCREMENTAL=0, CARGO_BUILD_JOBS=4, new empty kernel-out-a004 and
record_run --fresh-package --capture-artifact. This also compiles both P64
dense-E4 reference entries for the first time. No source mutation during build.

Immediately freeze executable, verify all captured source/config hashes, and
read its embedded registry. Compare candidate images to resource-generated
UP/GATE and DOWN images; report any unavailable/mismatching comparison
explicitly. Reference images are new and have no prior actual-device proof.

Root alone may run the future <=70s, no-argument diagnostic. Correctness is
all-output intended-code equality, not model accuracy; projection-only timing
excludes quantization/packing/GeGLU and cannot pass the complete speed gate.
