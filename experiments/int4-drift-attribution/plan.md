# Fixed-input drift attribution

Base: `1a835e18`; actual complete experiment323067, layer0 supplied weights and
synthetic512-row inputs. Matched step0 joined relativeL2=.5744822058 and final
residual=.8466086464 are observations, not model-accuracy results. Its complete
performance gate failed; this diagnostic makes no performance claim.

Preserve every existing device function, dependency, independent rational
quotient oracle, fixture and old complete probe. Add only a disconnected binary
and diagnostic CPU support. First reproduce the old matched step0 with separate
poisoned owners; do not use divergent step1 for attribution.

Exact boundaries before attribution:

1. Original P128 prefix/residual reference must match the candidate canonical
   BF16 prefix and persisted midpoint. Requantize that identical BF16 prefix
   with existing `ops_native_fp8_p512_blocks::quantize_hidden`; compare every
   code/scale with actual production `qi/xs`. No A4 is involved in this gate.
2. Independently reconstruct original F4/per16-E4 provided weights. Audit
   current native E4 blocked mapping, power2 row scales and finite nearest-even
   codes against this BF16 reconstruction; candidate I4 uses direct provided
   BF16/per960 codes and never multiplies native row_dequant a second time.
   Compare original reciprocal global values, retained immutable source bytes,
   and report complete weight reconstruction errors plus column ranges.
3. Existing reference projection with production E4 activations/weights,
   per960 block scales=1 and actual native row scales must be compared against
   actual current production output. The reference adds identity local
   multiplication and a F32 DM boundary: exactness is a required negative
   control, not an assumption. On mismatch report the mismatch and stop factor
   attribution; do not relabel it quantization error or relax tolerance.

UP/GATE factorial, all512 rows on the same canonical BF16 prefix:

| arm | activation codes/row scale | weight codes/group scale/row scale |
|---|---|---|
| EE | actual current E4 | actual current E4 / ones / native row_dequant |
| IE | exact rational A4 embedded exactly in E4 | current E4 / ones / native row_dequant |
| EI | actual current E4 | direct provided-BF16 I4 embedded E4 / per960 / ones |
| II | exact rational A4 embedded E4 | direct I4 embedded E4 / per960 / ones |

All use existing `ops_int4_group960_reference::project_upgate`. II must equal
the independently checked old candidate. Report UP and GATE separately and
combined, error versus EE and versus actual production, and an elementwise
factor interaction residual II-IE-EI+EE (not an additive decomposition of L2).
Run the unchanged original P128 GeGLU on each output with original globals;
retain its two BF16 global cuts, Erf, product/2 and final BF16 boundary.

DOWN factorial uses **production GeGLU BF16 hidden** as one fixed input for all
four arms, independent of UP/GATE arm. Obtain this from actual production joined
output through original P128 GeGLU, then check existing standalone E4 hidden
quantizer against production qh/hs. Use the same table with existing reference
`project_down`, original DOWN global and original BF16/global boundaries. EE
must equal actual production DOWN; II is a local DOWN approximation, not the
full candidate endpoint. Apply original post-FFN P128 tail to each output with
the same midpoint residual to separate tail propagation from upstream drift.

Report complete row reconstruction errors and zero fractions for E4/A4 versus
fixed BF16 inputs (H3840 and L15360), row/channel amax summaries, and full
provided-BF16 weight reconstruction errors for current E4 and candidate I4.
Do not reuse integer-only F64 dot bounds for mixed E4 factor arms; intended II
keeps its existing independent integer/F64 gate. Floating reduction and
rounding interactions remain explicitly separate from weight/activation
representation error.

Use original row order and its existing true512-row permutation. Check all
factor output permutations, reuse with separate poisoned outputs, and all
inputs/scales/weights for immutability. No timing comparisons: readbacks and
canonical BF16 conversions are diagnostic work. Existing current-production
graph remains the only performance reference; no new performance gate or
optimization is introduced.

Device images are reused, with no new #[device] function. A later fresh build
must bind every embedded callable to its existing actual/resource image or
explicitly mark any prior-unbound existing quantizer image; source reuse alone
is not actual image equivalence. This checkpoint permits source and a plain
host cargo check only, using the owned prefill target and original Cargo/lock.
