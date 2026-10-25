# Deterministic per960 scale fitting: plan before implementation

Control is fixed-F4 nearest-even quantization with the smallest power-of-two
scale `2^e` satisfying `6*2^e >= group_amax` (`8bf9789c`, packed-only058f785b).
The runtime fixed-F4 lookup/local-F32-scale ABI remains unchanged. E4 activations,
row dequantizers, original globals and all FFN rounding boundaries remain as in
the partial-scale candidate. This is another static weight approximation.

Candidate0 is exactly the control scale. The remaining24 are
`s=(m/8)*2^(e-d)`, `m=8..15`, `d=1,2,3`, in that fixed order. All are normal,
exactly representable dyadic F32 values. All-zero groups keep scale1/signs.
For each candidate use the same nearest fixed E2M1 value, ties to even magnitude
code and saturation to magnitude6; negative zero retains its sign bit.

Build a histogram of finite absolute E4 bytes. Each E4 value is integer `u/512`;
each F4 level is `h/2`, `h in {0,1,2,3,4,6,8,12}`. For a group with control
exponent e, choose common unit `q=2^min(-9,e-7)`. Then both source values and
every reconstructed candidate level are integers in units q:

```
source_integer = u * 2^(-9-min(-9,e-7))
level_integer  = h*m * 2^(e-d-4-min(-9,e-7))
SSE_integer    = sum(histogram[b] * (source_integer[b]-level_integer[code[b]])^2)
```

All shifts are nonnegative for e=-11..7 and d=0..3. Conservative global bound
`960*(448*2^18)^2 < 2^64`; actual valid groups have much tighter bounds. Thus
u64 arithmetic is exact. A strict lower-SSE comparison preserves the baseline
on ties. The minimum cannot have worse SSE against current E4 within any group.
Since retained row dequant is constant within a group, that guarantee also holds
after multiplying both reference and reconstruction by that row dequantizer.
It does not guarantee error against original NVFP4 or model/task accuracy.

Precompute code/error tables for19 exponents and25 choices. This avoids a
per-element nearest search but still costs up to25×127 histogram multiply-adds
per group, approximately28B for48×3 matrices. Measure one59M-value projection's
encode-only CPU time and bounded RSS before judging startup viability. There is
no assumption that the fitting step is free.

Gates: independent Python Fraction code/scale/SSE selection proof, no-worse-SSE
check against the frozen control, signs/zeros/thresholds/saturation and true
blocked row/K permutations. Then supplied-weight-only diagnostic for12
projections across layers0/5/6/47, reporting baseline/new current-E4 and
row-dequant-weighted L2 plus isolated CPU fitting time. No serving/default
change, no Arena dispatch by this agent, no quality inference from weight L2.
