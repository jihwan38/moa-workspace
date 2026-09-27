# CPU bound for a coupled native-down representation repair

Reproduce with `python3 moa-dev/experiments/candidates/ffn-native-down-weighted-residual/numerical_bound.py --check`.
The companion JSON records script/production/model source hashes, all counts,
worst cases and the exact limits below. This is a CPU RNE representation
calculation, not a new kernel, native-DPE test or whole-FFN acceptance result.

The production normalization scale is an exact F32 power of two. Finite BF16
inputs divided by it have at most eight significant binary bits before F32
underflow, which can only reduce that precision. The scale rule bounds
`|u| <= 32640`. We exhaust 20,224 nonnegative such F32 values, including
subnormal values down to 2^-149, and verify negative symmetry. This conservative
superset includes all18,176 nonnegative BF16 bit patterns <=32640.

For current `hi=E5(u), lo=E5(u-hi)`:

- Maximum `|u-hi|` is2048.
- Maximum `|u-F32(hi+lo)|` is128. For example u30848 gives hi32768,
  lo-2048 and third residual128.
- Multiplying only the last residual by256 gives a maximum cast input32768.
  E5M2 max finite is57344;512 would produce65536 and overflow.
- Third-only scaling fixes the known full-FFN stress construction but leaves
  maximum normalized absolute reconstruction error2^-21. Small inputs whose
  first two terms underflow then have only one E5 mantissa to encode them.

A better coupled representation keeps the same three payload terms:

```
hi = E5(u)
lo16 = E5(F32((u-hi)*16))
r3 = F32(u - F32(hi + F32(lo16/16)))
third256 = E5(F32(r3*256))
x_hat = s * (hi + lo16/16 + third256/256)
```

Exhaustive results: maximum first residual2048, maximum third residual128;
scaled cast inputs both at most32768. The largest safe power-of-two
multipliers are therefore16 and256 respectively. Maximum normalized absolute
error is **2^-25**,16 times smaller than third-only scaling. Every enumerated
value with `|u| >= 2^-17` reconstructs exactly; the largest positive inexact
value is `2^-17 - 2^-25`. Values sufficiently below that range can still round
or disappear: this is not an exact encoding of every BF16 value.

The known stress has x2^-22, cluster scale2^-4 and normalized u2^-18. Both
third-only and lo16/third256 reconstruct it exactly; both anchors and the other
cluster's tiny values also reconstruct exactly. The CPU full-FFN model then
predicts0.404296875 in both output halves, with zero modeled tolerance
failures. This is not an NPU or public PRNG-fixture result.

Production `down-e5-v2/mlp.rs:318–363` computes the terms, and413–425 currently
contracts `FfnTerm` away before common scale restoration. The coupled repair
must retain that axis, apply F32 coefficients `[s,s/16,s/256]` per term and
then sum. Its quantizer could insert lo's×16 in the existing pass and the
sum-terms pass's /16 before addition; that is a source-level feasibility
hypothesis, not proof that emitted engine count/lifetime is unchanged. Native
term output, coefficient VRF, F32 reduction and any changed partial storage
must all be included in exact graph and complete-FFN tests. A quantizer-only
edit would compute the wrong result.

The normalized error bound becomes s×2^-25 in activation units; it is not an
absolute bound on projected or RMS-normalized output error. Native summation
order, hardware cast/subnormal behavior, target lowering, unchanged numerical
acceptance and whole-kernel performance remain independent gates.

The script binds the E133 frozen production source and its source-bound public-contract CPU model by SHA256. The latter retrieves contract/reference files from source baseline `cc82b9ccff22795165faffd3ba9a68cc557da001`; E134 starts from record commit `543a947`. No current runtime or acceptance code is changed by this analysis.
