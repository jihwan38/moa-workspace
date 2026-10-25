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
