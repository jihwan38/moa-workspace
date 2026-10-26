# Actual323285: correctness PASS; complete10% speed gate FAIL

The retained-HBM three-graph diagnostic completed in14.698629789s with exit0.
All correctness and composition checks passed. It measured4.09–4.43% lower complete
latency than the exact current q020 control, but **both preregistered >=10% gates
failed**. Decision: **HOLD; no model integration or serving promotion**.
Exit0 reports correctness only. This experiment provides no model/task-quality,
HTTP throughput or official-score result.

| Row order | q020 whole graph µs | Three-graph µs | Gain |10% gate| Further removal needed µs |
|---|---:|---:|---:|---|---:|
|0|1061.571125|1014.5459375|4.4297727%|FAIL|59.131925|
|1|1057.9900625|1014.702250|4.0915141%|FAIL|62.51119375|

Each mean covers16 completed chains per arm:8 restored-input samples per block
in candidate–q020–q020–candidate order. All three awaits, checked raw binding,
retained canonical BF16/packed/scale/offset HBM boundaries and the original tail
are inside the timer. Resets, uploads/downloads, startup and independent oracles
are outside. The control is the unchanged one-call production grouped-store FFN,
with its exact previously validated embedded image.

## Exact composition and intended-code checks

- All11 scratch fields versus the separate seven-call chain:255,918,080 bytes.
- All11 fields versus normally typed three-call inputs:255,918,080 bytes.
- Mid-prefix persisted residual:15,728,640 bytes.
-190 negative owner validations rejected before any function lookup/launch.
- Typed correctness-only input copies remained immutable:19,660,800 bytes.
- Independent intended-code comparison events:196,608,000 BF16 values and6144
  sampled i64/F64 bounds; true permutation coverage78,684,160 mixed values.
-40,960 group scales,40,960 relative LUT offsets and19,660,800 packed bytes checked
  independently against the unchanged rational/packing oracles.
- Post-reuse equality:482,189,312 bytes across three independent candidate-style
  scratch sets and production scratch. Direct immutable checks599,335,048 bytes,
  plus the separately counted typed inputs above. Production weights/norms use a
  separate full-read FNV fingerprint over394,728,988 bytes, not byte-array equality.

Counts are repeated comparison events, not unique model logits or task answers.
All512 rows, four/16 activation groups, two row orders and two feedback steps are
included. Each row order uses supplied layer0 weights/norms/globals and synthetic
finite projected/residual tensors; full-attention prefix and padded tails are absent.

## Arithmetic and drift remain identical to grouped seven-call323213

All12 printed drift records,8 candidate reconstruction records and8 production
zero-code records match323213 **as complete strings**. Provided weight fingerprints
and group encodings also match. Matched-entry step0 relativeL2 remains:

| Output | RelativeL2 versus current E4 |
|---|---:|
|Joined UP/GATE|0.3963337423|
|DOWN|0.4820915604|
|Final residual|0.4238808191|

Step1 uses independently evolved residual feedback and is not a same-input error
isolation. Candidate prefix reconstructionL2 remains0.427417544056 with69.4473%
zero codes; its own GeGLU hidden reconstructionL2 is0.2895428266901 with84.4387%
zero codes. That hidden input already differs from production. It cannot be
compared directly with attribution323102's0.504361087 row-A4 error measured on
fixed **production** hidden. Scalar RN32 reconstruction is not a dot-product error
decomposition, and these synthetic drifts do not establish model/task accuracy.

## Separate same-job seven-versus-three diagnostic

| Row order | Seven-call µs | Three-call µs | Gain |
|---|---:|---:|---:|
|0|1081.7929375|1015.1664375|6.1588958%|
|1|1080.718375|1015.3340625|6.0500787%|

These are separate restored-input ABBA blocks,16 chains per arm. They show an
inclusive composition benefit in this job, but cannot replace the q020>=10% gate.
Their difference combines launch/runtime and compiler scheduling effects; it is
not four measured constant launch costs. No per-op device trace was requested or
returned, and no cache/frequency cause is inferred.

The eight separately instrumented completed-await repeats average:

| Phase | Mean µs | Range µs |
|---|---:|---:|
|A: prefix + H quant/pack|114.615625|112.812–116.892|
|B: UP/GATE + GeGLU + L quant/pack|605.982625|603.190–609.971|
|C: DOWN + original tail|297.395000|293.735–304.685|

Mean phase sum1017.993250µs; enclosing total1018.173375µs; clock gaps0.180125µs.
The diagnostic one-call control averages1098.910875µs, which is distinct from
matched q020 ABBA. These phase clocks cannot be substituted into the speed gate.
Tiny gaps are clock bookkeeping outside the stages, not measured launch overhead.

## Startup, storage and provenance

CPU provided-BF16 reconstruction/per960 encoding **plus audit**:5895.102297ms.
Original production load/prepack/join:62.233570ms. Candidate **plus intended E4
reference** uploads:217.812180ms. All are excluded from chain timing and preserved
as adoption costs; these are not candidate-only or48-layer startup estimates.

Declared retained payload subtotal901,981,728 bytes includes candidate scratch
63,979,520, production scratch49,156,096 and two extra parity scratch sets totaling
127,959,040. It excludes temporary oracle/typed input copies, CPU copies, allocator
and device DM/TRF. It is neither measured server peak nor a full-model fit proof.
All logical HBM boundaries remain present; compiler resource success does not
waive runtime/model-quality gates or the old ungrouped DOWN resource HOLD.

Build/submission provenance source ec833206 (functional d5e70b98), captured ELF
`13c27dfbe0b24e8ce6d926325b0657683c8c9970f46c6812f98c554da2add70b`,6,077,088 bytes.
All424 source/config hashes match build/submission/current files. All19 old images
match323213 and all3 new images match the frozen resource checkpoint; all22 match
the fresh build's embedded registry. The actual run was submitted while two
analysis files were untracked; no captured source/config input differed.

Raw log SHA256 `8668530e6f5a8ca91b597e3aff34a857a3e8fdb1672c80abd068b2c863eaa80d`.
Run/status/source/log copies and checksums are in actual-323285/. Replay:
`python3 experiments/int4-grouped-threegraph/analyze_actual.py --verify-local-artifacts`.
All16 timing blocks and8 phase repeats, both failed gates and all scopes are retained
in actual-result.json. No source change, new build, further Arena run or candidate
implementation was made while collecting this evidence.
