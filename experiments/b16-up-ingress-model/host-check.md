# Frozen host checks

Source c26ca9e plus probe288cead, base624431f. a004 unfiltered library tests
passed111/111, zero filtered/ignored, in0.30s after1m41s package build.
CARGO_INCREMENTAL=0, reused prefill target, package-only test debug0.
a005 plain server+compare_b16_up_ingress check passed12.88s. No features.

batch_kernels independently reviewed c26ca9e/288cead: both per-call arms
preserve Automatic wide/output/R32/O32/current attention composition;
only canonical all-singleton B16 with staged+composed epilogue uses the
candidate. No persistent selector/cache ownership changes. Disjoint paired
histories and active128 ownership remain valid; all48 quanta finish before
31-token prompt reuse, step8 disables FFN/O/QKV both arms and step9 reenters.
No source/control blocker found. This review and CPU check do not substitute
for whole-model/cache device gates.

source-equivalence.json verifies all five candidate/control norm/FFN device
modules are byte-identical to actual micro r1056. Kernel/probe/evidence source
is preserved in sibling b16-up-ingress; runtime job320194 exact and ~4%
complete epilogue conservative improvement. No serving activation here.
