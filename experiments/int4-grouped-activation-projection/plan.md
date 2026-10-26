# Grouped activation projection preregistration

Base a4152dba preserves actual 323067 and 323102. Existing complete I4 FFN is
3.5–4.3% slower than the current E4 graph; its >=10% complete gain gate failed.
Original DOWN resource 245684 > 240000 remains HOLD. This source/resource task
cannot promote a model or establish speed or task quality.

For each token p and 960 group g, finite BF16 inputs are FTZ-normalized and
A_pg=max(2^-112,max(abs(x))). The unchanged independent u128 quotient oracle
computes q=sign(x)*min(7,floor(7*abs(x)/A_pg+1/2));
sx_pg=RN32(A_pg*RN32(1/7)). Provided layer0 BF16(F4*E4) weights retain the
existing per960 rational encoding and sw_rg; original native-E4 row dequant
must not be reapplied. New intended arithmetic is

    d_pg,r = sum_960(qx_pg,k*qw_rg,k)          // exact I32, |d|<=47040
    t_pg,r = RN32(RN32(F32(d_pg,r)*sw_rg)*sx_pg)
    y_p,r = BF16(original F32 inter-slice Add_g(t_pg,r))

Weight Mul0 then activation Mul1 occur before reduction. The final vector
stage casts the sum directly to BF16; the old F32 DM/xscale pass disappears.
DOWN then preserves BF16 projection -> original reciprocal-global multiply
-> BF16. UP/GATE remain pre-global. This changes both activation quantization
and floating-point multiplication placement; no equality to old row-A4 math.

The packed input owner/wire remains I32 [P/256,K/960,P%256,K%960/8], publicly
bound to the same-byte typed I4 input. Group scale ABI is F32
[P/256,K/960,P%256], offset=((p/256)*(K/960)+g)*256+p%256:
H3840 uses8192B, L15360 uses32768B. The new raw adapter retains 5 UP/GATE and
6 DOWN argument slots, exact byte extents, 256-byte alignment, checked end,
output/input nonoverlap and borrowed owners through awaited public run.
Old 2048-byte row-scale owners must be rejected before lookup/launch.

Separate new dense-E4 reference starts from the frozen P64 reference geometry,
not the I4 P128 body. E4 represents every integer code -7..7 exactly; local
F32 integer dots <=47040 are exact. It implements both group multiplications
and the original reduction/global cuts. No candidate math helper is shared.
All BF16 output bits must match raw-I4, typed-I4 and dense-E4 arms.

Independent CPU bound uses i64 local dots and F64 sum(d*sw*sx), including its
own rounding allowance: d*sw is exact F64, but multiplying by sx and summing
need not be. For N=4/16, m=3*N+4 and gamma_m=m*2^-24/(1-m*2^-24), radius is
  gamma_m*sum_abs_terms + m*MIN_NORMAL*(N+sum_abs_sx)
  + 64*EPSILON_F64*sum_abs_terms.
This conservatively includes first-product FTZ amplified by sx, later FTZ,
and F64 center rounding. All intermediate/end-point finiteness is checked.
Round interval endpoints outward to F32 then BF16. DOWN global multiplies
both BF16 endpoints and takes min/max, retaining its second BF16 boundary.
512 preregistered token/output-row samples per projection/order are checked.

CPU inputs cover all512 tokens and every H4/L16 scale slot; each token/group
has a distinct normal BF16 maximum and row/group signed-code tags. A second
order is the true row bijection (73*p+127)%512. Every raw packed word is
independently reconstructed little-endian and compared to pair-nibble bytes.
Candidate-first separately poisoned outputs; all output coordinates, row
permutation, source/device scales/weights/inputs after timed reuse, and raw
negative owners are checked. CPU weight conversion/upload is startup only.
Any future projection ABBA includes the complete awaited raw call but excludes
quantization, packing, host conversion and download; it is not full FFN.

Potential removed cost: old separate scaling passes total47766 estimated
cycles (4 UP+2 DOWN). Added group-scale delivery is UP1048576B versus262144B,
DOWN524288B versus32768B, with another local multiplication. These are static
work quantities, not critical-path latency predictions. First gates: one host
check, resource compile UP/GATE and DOWN, exact full timeline and no DramStack.
No width/shape sweep. New source and reference will receive independent review
before root chooses any standalone build or actual execution. Batch owns the
separate grouped quantizer; this projection probe uploads CPU codes/scales.
