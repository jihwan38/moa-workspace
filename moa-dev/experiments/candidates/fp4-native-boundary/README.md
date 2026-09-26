**2026-09-27 후속:** [전송 계약 수정 및 E123 전체 FFN 실측](../ffn-direct-boundary-recompute/README.md). v7c에서 물려받은 joint 입력 DMA의 padding 범위 오류를 수정한 direct 경로가 독립122880행과 전체 FFN 기본 검사를 통과했다. E123 단회 이득은 동시 M03 대비0.7392%로 작아 미채택했다. 아래 합성 실패를 일반 native DPE 한계로 해석하지 않는다.

# D-FP4-NATIVE — locate the first incorrect direct-decode boundary

Branch `diag/fp4-native-boundary`, starting from integrated M04 record `c49300d`.
Adopted control remains E104. Latest user target is top10; no ranking gain is
claimed by this diagnostic. This is a bounded correctness investigation of the
previous C07 failure, not a renamed optimization family.

Prior C07 direct E5M2 decode/native dot variants failed full-shape projection
oracles, whereas FP8 materialization variants passed. Reload alone did not repair
the failure. The prior whole-projection outputs do not locate the first broken
boundary. M04 row tiling retained all FP8 bytes and regressed13.67%; no tile sweep.

Question: with identical full-shape packed weights and already representable
E5M2 input terms, does corruption first appear in FP4 lookup itself or in its
composition with native DPE/collection? Does a materialized control remain correct?

The first probe uses L15360×H3840, row ownership l=7680c+30s+r,
three E5M2 terms in TRF and the same64-wide native contraction/block16 outputs as
E104. It intentionally excludes BF16 input quantization, input restoration scale,
block scale and final row reduction so their errors cannot mask the raw boundary.

- `decode`: FP4 HBM -> FP4 DM -> E5M2 lookup -> full E5M2 DM/HBM output;
  compare all58,982,400 values with independently specified sixteen FP4 values.
- `materialized`: identical lookup -> E5M2 DM -> native dot -> F32 block partials
  -> HBM; compare all3,686,400 raw block results per input case.
- `fused`: FP4 DM -> lookup -> same native dot without E5M2 DM boundary -> same
  F32 block outputs. Independent process/device from other probes.

Two exactly representable input cases cover constant distinct terms and varying
signed terms/column positions. Each block result is accumulated in independent
host F64 and asserted exactly representable in F32. All16 FP4 codes occur.
Any NaN, Inf or output mismatch is diagnostic failure, not a valid speed result.
The host reports row/block samples and finite/non-finite counts.

Extra output DM/HBM boundaries, controlled FP8 inputs and separate images make
these diagnostic programs unsuitable for performance/adoption claims. No free
host weight transformation is proposed for a submitted kernel. A successful
repair must return to the unchanged public inputs and include all preparation.

First gate: compare the three boundaries on Arena, then choose a concrete source
or packet/layout repair based on the failing values/lowered configuration. Do not
label an unresolved mismatch an SDK bug. Preserve source/recipe/log/hash before
each attempt. Once the primitive is correct, validate the complete projection
and FFN with contemporary controls, then expanded correctness before adoption.

Source arithmetic/layout origin: E104
`7f891f9c48e670fe014b2ac5bd2c92bae3bac433:src/device/shared/mlp.rs`.
No competing source is used. The public SDK supports lookup on Main; it is not
moved to Sub. All diagnostic files remain in this directory.

## Boundary results — closed unresolved, no adoption

- v1 / job103972: isolated materialized and fused native dots pass two cases
  (7,372,800 F32 block values each). Decode host failed `Unplaced` before launch
  because the output was not allocated; v2 fixes that source error.
- v2 / job103997: decode58,982,400 values and both dot+identity-VE paths across
  four magnitude cases pass. Identity immediate does not test a runtime VRF read.
- v3 / job104015: a real HBM->VRF restoration scale (1.25 or2^-16) is included;
  all four cases of both raw-block paths pass.
- Immutable old-image replay104011: old materialized C07 probe passes207360
  outputs, old direct probe still produces mismatches and a case2 timeout.
  The exact old binaries are verified; changed runtime alone did not repair it.
- v4 uses real E104 input preparation; initial compile used an unpadded `Slice`
  type, repaired to the actual m![1 #256] alias in v4b.
- v4b / job104040: fused path passes14,745,600 raw block outputs from actual
  BF16 preparation. Materialized debug-output path times out before its first
  result, so this job is not an all-passing paired control. The added large
  output boundary changes scheduling/lifetime; its timeout remains unresolved.
- v5 / job104091: direct complete single projection passes61440 F32 rows across4 cases. Materialized control again times out; no passing paired-control claim.
- v6 / job104117: two direct projections share prepared TRF/scale and each output F32 rows before gather/cast. Nonzero cases fail for almost every row (samples are zero); zero input passes. This reproduces corruption before final BF16 conversion. Materialized control times out.
- v7/v7b: missing Rust generic arguments, then interleaved-axis lowering mismatch.
  v7c uses two HBM transfers into tiled views of one packed Projection-axis DM
  tensor; no host prepacking or extra full DM copy.
- v7c / job104178: joint lookup/native producer reduces failures to16 rows in
  projection0 (local row0/slices0..7 in each cluster), while projection1 passes
  all cases. Zero-input case passes both. It is still incorrect and not adopted.
  The same spatial pattern occurred in the old QKV native-pair diagnostic; that
  correspondence is an observation, not proof of a compiler/hardware cause.
- v8 moves restoration scale from the joint DPE task into the existing F32
  reduction stream, before block-scale multiplication. It keeps arithmetic
  order and partial storage size, testing the native-DPE/VE composition boundary.
  Job104197 fails both projections in all three nonzero cases, mostly non-finite;
  the zero case passes. This did not repair the boundary.
- v9/v9b prefix-row source attempts failed Rust ownership and source tile-axis
  checks. Corrected v9c (32 physical rows) and v9d (36 physical rows) both reached
  the backend assertion `stride must divide the size`. The attempted source
  padding has not been demonstrated executable; this is not an NPU result.
  v9t/v9dt are unbuilt trailing-padding source sketches, not evaluated variants.
- v10 / job104243 adds an entire duplicate up projection before the two real
  projections in the joint stream. Both real projections produce non-finite
  outputs in all four cases, including zero input. Full-projection warmup is not
  a repair. Its extra weight load/dot/partial is explicitly diagnostic overhead.

The materialized debug-output control continues to time out in these later jobs;
none is an all-passing paired screen. The exact older materialized control passes
and E104 remains adopted. No new complete-FFN performance measurement or official
submission was made. One projection's success does not validate composition of
two projections, nor establish a compiler or hardware defect.

The bounded investigation is now **on hold with unresolved composition errors**.
Further padding/setup sweeps lack a demonstrated repair mechanism. M03 uses the
validated FP8 materialization boundary and changes whole-FFN ownership instead.
Any future direct-fusion retry must identify a supported state/layout change and
validate the two-projection path before progressing to performance experiments.
See `decision.json`, `verification.json` and immutable per-attempt evidence.
