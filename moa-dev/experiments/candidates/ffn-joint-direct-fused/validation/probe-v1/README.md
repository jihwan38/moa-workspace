# Joint direct fused output diagnostic

This diagnostic calls the exact candidate's `prepare_ffn_fp8_split` and
`project_up_and_gate`, including joint direct FP4 streams, the first BF16
cast and local F32 gather to the GeGLU consumer layout. It observes both 15,360-row outputs before GeGLU or normalization can hide
errors. The candidate `mlp.rs` is copied as a full-file byte-identical prefix;
only a child module is appended to access its private helpers. The separate
root `#[device]` entry reuses M03 v2's fix for image registration paths.

Eight finite dyadic cases check245,760 outputs. Cases0..5 reuse M03's independently
specified FP4/F64 oracle: all16 nibble codes and row identity, sparse block
identity, cancellation across16-element blocks, zero, and two mixed-tiny inputs.
Cases6/7 add K15/16,63/64,3839 coverage and row6/30/cluster-dependent block scales.
Up and gate have different weights and scales for0/1/3/6/7; their scale buffers
are intentionally equal to1 for cancellation/tiny cases2/4/5. Every row is checked,
including5/6,59/60,119/120,7679/7680 and15359. Case7 changes runtime restoration scale to2^-7.

`quantizer_audit.py` independently enumerates finite E5M2 values and rounds ties
to even. It confirms the production quantizer formula restores every distinct
input value in these cases. The host asserts actual BF16 inputs equal the intended
F64 values, calculates each FP4×activation block and scaled sum in F64, checks
its chosen block/final F32 exactness, and obtains final BF16 with the installed
host scalar conversion. These assertions alone do not establish all possible
intermediate F32 sum orders; actual NPU comparisons remain required. This is
neither all-BF16 correctness nor a fix for the inherited3-term approximation.

All output buffers start as NaN. Comparison is BF16 bitwise except signed-zero
equivalence; any nonfinite output fails. Deterministic FNV checksums of the actual
packed weight, scale values encoded as F32, activation and expected output buffers must match between
the CPU-only run and Arena. They detect changed inputs; they are not cryptographic
proof of the oracle. SHA256 separately pins sources, image, binary and actual
uploaded archive. This diagnostic image and its timing are excluded from every
whole-FFN performance comparison. Down, GeGLU and the epilogue are checked through
the separate whole-FFN suite.

`../../probe.py build` performs the exact compile/host build and CPU-only oracle.
It uses the idle warmed `/tmp/moa-m03-validation-probe/target`, agreed with root,
and a fresh `/tmp/moa-e131-validation-probe/e131-joint-probe-v1` work directory.
`submit` requires that build's checks; `collect` preserves raw logs/archive and
both success/failure; `--check` re-verifies all saved evidence without compiling
or submitting. Root owns adoption and further research decisions; this probe
contains no performance threshold or automatic sub5% rejection.

The corrected E128 host/oracle is reused. Production source binding is filled only after a frozen whole-kernel build is selected; preparation is not a correctness result.
