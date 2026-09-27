# E134 actual-down oracle — job 109958

[Results](results.json), [source/image build binding](build.json), and
[raw log](evidence/arena-109958.log) preserve the run. All **9 cases / 34,560
BF16 outputs passed exactly** (signed zero equivalent); maximum absolute error
was zero. The job succeeded with exit 0 in 3.327 seconds, and `probe.py --check`
recomputes the source, embedded image, original inputs/reference and actual
uploaded archive checks.

The actual frozen `weighted-v3/mlp.rs` prefix calls `project_down_tile` and
`finish_down`, including cluster-local quantization, three-term packing and
broadcast, native dot, coefficient restoration before F32 term reduction,
block scales, column sums and cross-cluster sum. The independent host reference
uses the original BF16 activation and public FP4/FP8 values; it never reconstructs
an E5 oracle. Every reference accumulation boundary is exactly F32-representable
for these selected cases, and all nine additionally require exact BF16 output.
Original comparison tolerances remain unchanged.

Cases 0–7 use exactly the same input/weight/scale/reference bytes as the prior
E133 oracle. E134 repairs its previously failing case 4: all 3,840 rows now equal
1.40625, versus E133's 0.703125. New case 8 uses per-cluster anchors 1024 and 0.5,
tiny values 2^-22, zero anchor weights, FP4 weight 6 and blockscale 15. Each output
half isolates one cluster, and all 3,840 rows equal the original-BF16 reference
0.1650390625. The case is tied to the separately constructed public full-FFN
stress input; **this oracle itself executes only the complete down helper**.

The test also covers differing cluster scales, K16/64/480/1920/7680/end
boundaries, zero, representable tiny values, and block/cluster cancellation.
These nine exact cases are not a proof of arbitrary BF16 equivalence. The
[CPU bound](../numerical-bound.md) still permits normalized representation error
up to 2^-25; multiplied by the cluster scale this is an activation-error bound,
not a whole-FFN output-error guarantee. The separate full-FFN stress report and
standard-input validation determine their own scope.

Production source SHA256: `a80ebfea24f029b039ae1b5ad5323770a1d414aee2d60640265c3e1c32d84b47`.
Diagnostic image SHA256: `a7a4080929da7a05717e8c06deae968c4f00492068cbfa842b289625f92444db`.
Actual uploaded archive SHA256: `bb2c4d7f693a7686623b11dcac818428bd91797047bcc06d09c9b5ec379137b7`.
No performance or official-submission claim is made by this oracle. The earlier
[weighted-v1 compile-only failure](feasibility/record.json) remains separate;
its source and evidence were not overwritten by the final v3 oracle.
