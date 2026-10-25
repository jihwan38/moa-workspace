Functional source3a877b1e9fa807d68515d2231ffc37c2a60e6f48; kernelc804b4d5.
Hostcheck passes. Independent scalar nearest-even E4 oracle passed103,686 finite
BF16/power2 checks, including signed zeros. One host test passed, no filter.

Recorded release build a001-b16-lm-fp8-build passed in38.51s. All393 source/config
hashes still matched the pre-build record before copying the immutable ELF.
Artifact and full provenance are in artifact-manifest.json. Root alone dispatches;
no arguments, FURIOSA_OPT_PROFILE=trace, timeout70seconds. No NPU result yet.

Accuracy scope: all1,006,632,960 provided embedding codes and262,144 scales are
checked independently, plus each normalized B16 input. Six runs cover three
non-dyadic fixtures and their actual row permutations; all25,165,824 output values
are compared against current exact BF16 storegroup. All384 greedy selections
(empty/default/winner-banned/all-banned) are reported. Output differences or changed
selections are approximation statistics, not equality passes. Independent intended
quantized F64 algorithm, finiteness, permutations and immutable source tables are
mandatory gates. The older H960 BF16 strict failure remains untouched.

Timing includes the extra input quantizer launch and complete native LM with both
scale multiplications and canonical output. Both arms start after actual final RMS;
RMS is unchanged. Static shadow creation/allocation time and actual memory delta are
reported separately. Standalone capacity is not full-model headroom validation.
No model, Worker, cache, selection or serving-default files changed.

Independent e2e read-only review of frozen3a877b1e passed mapping, quant scale
strides, intended algorithm and oracle/timing scope; no concrete blocker. Root
submitted r1184/job321121 with trace/70s. Result is pending; submission is not a gate.
