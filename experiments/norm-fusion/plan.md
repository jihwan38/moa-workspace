# Exact normalization and quantization fusion

Base6d58608. Candidate replaces four post_attention_p128 calls, four
rmsnorm_p128 calls, pack_hidden_group, and quantize_hidden with one kernel.
Input projections and each request's residual remain separately owned.
The original H/480 reduction, post-normalization BF16 cast, residual BF16 cast,
pre-normalization BF16 cast and established full-row FP8 quantizer are reused.
Residual x is still written to HBM for the later FFN residual/yield checkpoint.
Only intermediate normalized and packed BF16 HBM buffers disappear.

Expected removal:5 full P512*H BF16 transfers=18.75MiB/layer,900MiB over48layers,
and9 host launches/layer. This is removed work, not a predicted latency win.
Potential added cost is DM redistribution from H480 shards to full-row quantizer
and writes into P512H960 blocked output from four P128 quarter mappings.

The first gate is an isolated complete10-launch vs1-launch ABBA probe, independent
poison outputs, all512rows exact FP8/scales/residual, zero/tiny/signed inputs,
three seeded quarter permutations and independent F64 residual checks. No server
caller changes or default adoption before root's actual NPU gate. Keep all failed
builds and reject if mapping/parity fails or complete-chain time does not improve.

Build a023 failed before execution: selecting one P512/256 half introduces skipped outer padding on H/960; the explicit requested tile omitted that skipped-padding marker. The repair retains H/960#{!}8 in both nested views; arithmetic and probe remain unchanged.

Build a024 rejected the explicit H/960 padding as duplicated extent (4MiB output for2MiB allocation). The skipped marker belongs on the selected P512/256=1#{!}2 factor itself, not another H/960 factor; a025 repairs that metadata only.

Build a025 passed in28.44s with skipped padding placed on P512/256=1#{!}2. The probe is ready for root-owned NPU execution; compilation is not a numerical or performance gate. Parent branch host cargo check had already passed. Independent source review by batch_kernels found no concrete arithmetic/quarter-mapping defect.

## Actual post-attention fusion microkernel gate

Root Arena318702, runs/r570-norm-fused-submit/arena.log, passed all3seeded
quarter permutations: each1,966,080 FP8 bytes,512 FP32 scales and1,966,080
persisted BF16 residual values matched original kernels exactly. Independent
F64 residual NRMSE was0,1.245e-5,6.247e-6. Complete10-launch controls were
405.9–427.1us; fused1-launch samples263.3–291.7us. The improvement is measured
only for this complete microkernel chain, not yet a whole-model throughput gain.

## Second bounded QKV ingress fusion

normalize_qkv_four removes the normalized BF16 write+read and seven launches
from four RMS+quantizer calls. It reuses the same H480 normalizer and full-row
quantizer, returning four existing row-major FP8 inputs/padded scales. Build
a026 passed32.29s. The independent candidate-first poison oracle compares all
1,966,080 bytes and512 live scales for3seeded/permuted groups; padded scalar
lanes are explicitly skipped because their values have no contract. Complete
8vs1-launch ABBA timing decides adoption. This second path has no NPU gate yet.

## QKV ingress gate and isolated host integration

Root Arena318707/r574 passed all512rows/scales and independent poison allocations
for three seeded permutations. Complete original8-launch chain was about289–294us;
fused1-launch path was53–76us. This validates the second exact fusion at kernel
scope. No activation precision or RMS association changes were introduced.

Host integration is isolated in norm-fusion-server, rooted at aef145a and updated
with fad6054's exact full-QK ingress. PreparedGroup owns one fused_norms choice;
normal constructors enable both proven fusions. Explicit original-norm diagnostic
constructors retain the previous launch paths only for the just-prepared state.
They perform normal allocation/alias checks and contain noawait after selecting
that state. Cancellation retains the existing discard-Model/Workspace policy.

compare_norm_fused compares original grouped Prompt512 versus fused grouped
Prompt512 with disjoint canonical caches, reversed physical pages, optional
48same-arithmetic R64 interleaves and four teacher-forced postprompt decode steps.
Timing excludes host logit reads for both paths and records stage profiles.
Existing compare_prompt_group also checks four independent request owners.
Host tests before QKV call adoption passed62/62; final checks follow.

Final combined host suite a029 passed62/62. a028 is preserved as an immediate unresolved-lib-export merge-marker failure before compilation; only both module exports were retained to repair it. No runtime behavior changed in that repair.

## Direct blocked post-FFN consumer

Whole-model318719 measured remaining FFN unpack1.980ms and postnorm8.347ms over
48layers. The bounded next prototype reads each H240-blocked producer quarter
directly into original H480 RMS shards, then reuses original normalize, residual
add and gate with every BF16 boundary unchanged. All four x checkpoints persist.
It removes two BF16[P512,H] transfers (360MiB/48layers) and four launches/layer.
The gate independently packs blocked bytes on CPU, compares every residual bit
against original unpack+fourpostnorm, checks immutable input, F64 samples, signed
layer gates and complete5vs1 ABBA. No host/default changes before its actual gate.

Direct blocked postFFN candidate a031 compiled in26.66s on first attempt. The producer H240 quarters read directly into H480 shards; the original norm/residual/gate helpers are reused. Actual parity/performance remains pending root.
