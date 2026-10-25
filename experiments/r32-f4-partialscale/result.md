# R32 direct F4/group960 resource gate: PASS

Base07718f0739a16f439c3725e31cb61ba4bcd480d5; candidate1a4a57a3.
Only new `ops_decode32_f4_partialscale` plus its lib export is functional.
`ops_decode32_ffn`, `ops_norm_r64` and every host caller are unchanged.
The seven copied activation/GeGLU helpers are hash-identical in
`unchanged-source.json`; e2e_evidence independently reviewed the scale axes,
F32 boundary and first32-row stores without finding a source mapping blocker.

| Existing launch | Current estimated cycles | F4 candidate |
|---|---:|---:|
|Pre-RMS, ops_norm_r64::rmsnorm_r64|23,662|23,662|
|Complete four-stage FFN core|283,759|213,075|
|Post-RMS/residual/gate|30,315|30,315|
|Sequential sum|337,736|267,052|

Core estimate decreases24.91%; including the unchanged surrounding norms,
20.93%. All four exact-resource compilations passed with no DramStack and
all source hashes unchanged. This meets the >=10% resource prerequisite.
The sum models the current three separate device calls, not a newly fused
graph. Host submission costs and actual device latency are unmeasured.

The complete core still loads/quantizes normalized input, computes UP/GATE
and original GeGLU, stores/quantizes hidden, then computes/stores DOWN.
HBM->DM operator count21→27 and summed estimates205,006→136,894;
TU33→36 and sums49,317→58,233; three new table-load spans sum3,879.
HBM output stores6/46,647 and internal DM copies2/1,126 are unchanged.
Source-identified weight-load estimates are49,157 each in the control,
24,612/24,612/24,708 in the candidate, with three additional group-scale
loads971 each. These overlapping operator sums are attribution, not latency.

Each matrix has58,982,400 weights. Total weight payload176,947,200 E4 bytes
becomes88,473,600 F4 bytes plus737,280 bytes of group F32 metadata. There
is no expanded E4 HBM bridge or runtime repacking. Input remains current E4,
TRF is32x960 bytes and each current L120/H120 weight tile becomes57,600B.
The original B16 F4 UpWeight/DownWeight and group metadata shapes can be
shared; no separate R32 shadow layout or persistent weight duplication is
required by this device ABI. Startup ownership/allocation is not implemented.

The scale is applied to every local K960 F32 dot before the original H4/L16
sum. A real reduced-F32 DM buffer then feeds the unchanged activation-scale,
retained weight-row scale, BF16 dot boundary and original global scale.
This implements newly quantized weights, so equality to the current E4 model
is not expected. No claim about model/task quality or C17 service latency.

Next gate, if selected: a fresh R32 intended dense-code+same-group-scale
reference, supplied layers0/5/6/47, independent F64 partial/final checks,
all retained scratch bits, distinct/permuted32 rows, input/output NaN tail
poison and immutable weights; inclusive current three-launch FFN timing.
No NPU job, host/default activation, micro executable or geometry sweep
was performed in this resource-only task. Reusable tail256 target is free.
