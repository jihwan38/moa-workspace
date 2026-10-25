# Layer0 intended-code I4 projection gate (source only)

Base/candidate: `0f053b7f7be14ef0a786ada4e59f7c307625a192`.
The official q020 source, model dispatch and dependencies are untouched.
UP/GATE resource356,610 passes the480,000 screen; DOWN245,684 exceeds240,000
and remains HOLD. Both reported noDramStack. Numerical success cannot override
that screen or establish a complete-path speed or task-quality result.

## Reference and intended arithmetic

The separate dense-E4 reference extracts the **unchanged** `stage_upgate` and
`stage_down` functions from `p512-f4-partialscale-micro` commit7dac9d0c.
`reference-source.json` pins full source and exact function hashes. It retains
P64 tiles (candidateP128), denseE4 contraction, public F32 scaled-partial
reduction and original output stores. The reference's additional row-dequant
input is exactly1.0. It must never receive the old native-E4 row dequant.
The extra identity multiply is disclosed in reference timings.

Provided layer0 UP/GATE/DOWN are reconstructed as
`BF16(E2M1(weight_packed nibble) * E4M3(weight_scale per16))`, before global
scaling, with strict tensor dtype/shape checks. Each canonical weight row is
quantized independently per960 by the unchanged u128 quotient/remainder oracle
in `support/rational_a4_oracle.rs`. Activation fixtures are separately quantized
per fullH3840/fullL15360 row by that same independent CPU definition. This
probe does not call the new LUT quantizer or device packer.

All codes are in[-7,7], exactly representable in both signedI4 and E4.
Each local dot is an exact integer bounded by960*49=47,040. Both paths then
multiply that local dot by its storedF32 per960 scale, reduce4/16 F32 shards,
multiply the activation row scale and round once toBF16. DOWN additionally
multiplies the provided reciprocal-global scale **after** that BF16 boundary
and rounds toBF16 again. UP/GATE global scaling and GeGLU are outside this
projection-only gate. No original-BF16 numerical parity or quality is claimed.

## Independent checks before any performance interpretation

- Candidate first, separate NaN-filled outputs; every joinedUP/GATE and DOWN
  BF16 value must be finite and bit-identical to the separate E4 reference.
- Two rounds: all512 deterministic non-dyadic BF16 rows, then the exact row
  bijection `(73*p+127)%512`. Every row has a unique nine-bit sign tag plus a
  row-specific amax. Both candidate and reference must reproduce the original
  outputs at the mapped rows. No response/data-conditioned resampling.
- Weight and activation packers independently verify every signed nibble and
  denseE4 code. Every physical group index is written once, with full coverage.
- For each projection/round,512 preregistered samples use token `p=i` and
  output channel `(317*i+19)%rows`, with canonical CPU indexing independent of
  the blocked device layout. Local dots are accumulated in i64.
- Let N be4/16, u=2^-24, g=(2N+2)u/(1-(2N+2)u). The F64 center is the sum of
  exact-in-F64 localdot*storedF32scale products, multiplied by storedF32 xscale.
  Bound is `g*sum_abs*abs(xscale)` plus
  `(2N+2)*MIN_NORMAL_F32*(1+abs(xscale))` for FTZ and a separate conservative
  F64 summation/multiplication error allowance. All relevant values/bounds must
  remain finite; overflow is outside the gate. The interval is widened outward
  when converted toF32, then monotonically rounded toBF16. DOWN propagates both
  BF16 endpoints through F32 global multiply and BF16 round, taking min/max
  even for negative global. This bound does not assume a reduction tree.
  Strict all-output E4 equality remains mandatory; the bound cannot excuse it.
- All supplied packed/scales/global source tensors are fingerprinted before
  reconstruction and after the last call. Every device input, weight, scale and
  unity reference scale is read back after timing/profile and compared in full.
  Fingerprints are labeled FNV1a64, not cryptographic source identities.

## Physical index proof

For canonical weight(row,k), parts=cols/960:
`g=((row/240)*parts+k/960)*240+row%240`, element=`g*960+k%960`.
This is [row/240,k/960,row%240,k%960]. Joining UP then GATE adds a leading
projection plane of58,982,400 elements. Group scales use the identical g.
For activation(p,k), group=`((p/256)*parts+k/960)*256+p%256`, then inner k%960.
For output(p,row), index=`((p/256*(rows/240)+row/240)*256+p%256)*240+row%240`.
The probe verifies group bijections and inverse coordinate mappings; per-group
inner coordinates are contiguous full960/240 ranges, with no padding bytes.

## Timing and scope

Layer0 only. Report host reconstruction/quantization, upload separately.
Completed awaited projection calls use fixed ABBA blocks of8 calls, both
UP/GATE and DOWN, including all real output stores; optional profiling is
outside all timing. Downloads, CPU encoding, allocation and upload are excluded
from these projection timings and labeled. There is no activation-quantize,
device-pack, GeGLU, completeFFN, model or HTTP claim. Actual root job budget70s
remains a future empirical gate; no runtime promise is made from resource counts.

At maximum simultaneous device residency: candidateI4+denseE4 weights
265,420,800B; group scales737,280B; two-arm outputs70,778,880B;
I4+E4 inputs14,745,600B; input scales4,096B; unity scales138,240B; global4B:
**351,824,900B (~335.5MiB)**, excluding allocator/compiler runtime overhead.
CPU also retains independent encodings and comparison/permutation snapshots;
provided mmap touched pages and upload/download temporaries can lift process
RSS above0.5GiB. These are accounting estimates, not measured allocator/RSS.

Current authorization is source-only: no check, build, test or Arena executed.
Future checks reuse only `/round2/ffn-k256/target`; future device compilation
requires its own fresh public `FURIOSA_OPT_OUT_DIR`. Root owns dispatch.
