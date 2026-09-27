# E140 quantization-boundary diagnostic preparation

**NotBuilt / NotTested.** Source-only wrapper; no compiled image, host, fixture,
submission or result exists. Production `local-v4/mlp.rs` is unchanged. The raw
production down oracle Arena111393 subsequently passed all9 cases/34560 exact
BF16 outputs, so repeating those dyadic inputs in this smaller helper was **not
selected**. Generic quant9 compile/NPU is explicitly on hold.

The remaining discriminating question is whether actual standard post-GeGLU
BF16 values reproduce a quantization error outside the complete FFN lifetime
context. A selected future run must obtain those actual input bytes first,
record their producing source/image or exact CPU-reference provenance, and use
the same bytes in the candidate quantizer and independent numerical reference.
Do not silently substitute the currently passing raw9 inputs or call a CPU
post-GeGLU tensor the actual device tensor. A CPU tensor can be used as a separate
identified numerical diagnostic if root selects it.

`probe_module.rs` is designed to append after the **entire unchanged production
file** and directly call `prepare_down_fp8_split` (local-v4 lines259–395). It
loads canonical BF16 L15360 as `[Dummy2,TileL7680]`, invokes the actual input
replication/K4max/chunked weighted quantizer, and exports its two return tensors.
No replacement quantizer, native contraction, RMS or performance measurement is
included. Exact lowering of these additional output stores is unverified.

Terms retain `[c,Hgroup64,q4,Term3,u1920]`, with flat offset
`((((c*64+g)*4+q)*3+t)*1920+u)`. All2949120 E5 values are observable, including
all64 otherwise-equivalent H groups. Coefficients retain
`[c,Hgroup64,q4,Term3,1#8]`:49152 physical bytes. Only each term's first F32 lane
is meaningful; the other seven are padding, must not be asserted initialized or
counted as checked outputs. Expected valid coefficients are `[s,s/16,s/256]`,
using the actual cluster7680 amax, floor2^-112 and exponent-derived scale.

A future independent host should model each F32 rounding step and E5 RNE, check
all valid terms/coefficients exactly (signed zeros may be reported separately),
and group mismatches by `(cluster,Hgroup,q,chunk=u/480,term)`. It should also check
that the64 Hgroup replicas agree for each identical input and that allfour
columns receive the same scale for that cluster. The input and expected-output
hashes must be checked before and after NPU execution. No tolerance change.

If this isolated quantizer fails on actual standard values, its output exposes
which scale/term/chunk/replica first diverges. If it passes, that does **not** prove
the whole FFN's quantizer state correct: removing native/epilogue consumers
changes resource lifetimes and scheduling. The next conclusion must be tied to
actual whole-context emitted register/dependency evidence. A passing smaller
helper cannot override the existing whole-FFN failures.
