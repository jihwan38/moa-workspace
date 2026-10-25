# BEEF Stage 2 objective

The active user-authorized goal is a verified final top-3 finish in MOA 2026 Stage 2.
Deadline: 2026-10-25 23:59 AoE = 2026-10-26 20:59 KST.
The final repository is `micro2026-moa/stage2-final-beef`; the root agent coordinates
official submissions and final-repository updates. Current official source/control
starts at upstream `8faa2422caa34c5742b89043b42d8d2ebf0c5b5c` (SDK 0.9.0).
Historical E154 and its evidence remain in `/home/jihwan38/moa-workspace`.

## Work organization

- `server/`: root agent's integrated, measured submission candidate.
- `native-ffn/`: root's FP8/native prompt integration, with frozen candidates for every evaluation.
- `r64-io/`: root-owned R64/B16 dense, embedding and vocabulary-head experiments.
- `r64-server/`: root's c009 + canonical64/R64 integration and throughput candidates.
- `prefill-nohistory/`: root isolated exact first128 causal attention pruning; no host activation until micro/model gates.
- `media-vision/`: root exact independent B16 vision patch encoders and guarded preparation batching.
- `e154-port/`: E154 migration and correctness control.
- `prefill/`: prompt/batch host execution work.
- `batch-kernels/`: reusable batched kernel feasibility and implementation.
- `ffn-k256/`: round2_path isolated K240 padded K256 / P256-TRF shared up-gate and matching column-major GeGLU experiment from eefe824; no serving or NPU dispatch.
- `ffn-shared-h960/`: batch_kernels agent exact shared-H960 up/gate utilization audit from2bbec7c.
- `e4-server/`: root sole-E4 KV integration; plain src-only production activation.
- `decode-blocked-server/`: batch_kernels agent exact decode layout integration and model oracle.
- `h480-server/`: batch_kernels agent isolated H480 FFN model/quality integration from3105678.
- `r64-ffn-fused-server/`: batch_kernels agent exact single-launch R64 FFN integration from30038f5.
- `p512-core-fused-server/`: batch_kernels agent exact single-launch P512 FFN core and dedicated384+128-owner constructor from4f0be45.
- `r64-prefix768-server/`, `r64-prefix768/`: e2e_evidence exact causal-prefix and attention movement probes.
- `norm-blocked-residual-host/`: e2e_evidence scoped persistent residual host adoption; consumer micro tree remains frozen.
- `native-attention-qk/`: e2e_evidence isolated native E4 QK attention over existing compressed KV; no production host/storage port.
- `prefill-head-attn-commit/`: e2e_evidence isolated exact blocked P128 heads, attention and KV commit fusion; retains Q in DM and keeps K/V scratch, no serving changes before complete-chain gates.
- `norm-blocked-residual-server/`: e2e_evidence exact persistent residual consumer chain and scoped host integration fromce88461.
- `norm-blocked-residual/`: e2e_evidence persistent H480 residual movement micro from4f7fe8f.
- `norm-quarter-post/`: e2e_evidence isolated exact grouped post-FFN quarter normalization fromc53ae8a.
- `r64-headrows-server/`: e2e_evidence isolated full-window HeadRows integration from4f0be45.
- `mixed-prefill-decode/`: round2_path unified mixed prompt/decode diagnostic worktree.
- `startup-trace/`: round2_path opt-in completed prompt chunk diagnostics.
- `startup-cohort/`: round2_path agent bounded first-output cohort scheduling experiment.
- `startup-rearm/`: round2_path agent conservative rearming experiment from c040, with the 3250ms receipt budget retained.
- `mixed-prefill-decode/`: round2_path agent isolated shared P128 prompt+decode diagnostic; no production scheduler changes.
- `startup-trace/`: round2_path agent opt-in per-chunk packing diagnostics; scheduler policy remains unchanged.
- `b16-greedy/`: round2_path agent exact B16 device token selection and retained-logit integration from 30038f5.
- `norm-fusion-server/`: round2_path agent isolated exact fusion host integration from aef145a.
- `norm-fusion/`: round2_path agent exact post-attention/residual/pre-FFN/quantization fusion from 6d58608.
- `media-audio/`: round2_path exact batched audio encoder, runtime helper and independent probe from7d6b102; root owns serving integration.
- `output-native-blocked/`: round2_path fused per-K FP8 activation and H64 blocked O prototype from640b167; no serving dispatch.
- `native-output/`: round2_path agent output projection utilization/precision experiments.
- `kv-profile/`, `kv-pool/`: e2e_evidence storage/attention profiling and kernels.
- `final/`: official private final-submission repository.
- `runs/`, `state/`, `tools/`: execution logs, resumable status, and orchestration.

Agents own their assigned worktree, commit coherent reviewed changes, and report
commit IDs for integration. Do not edit another agent's worktree. Coordinate NPU
jobs with the root agent; only root makes quota-limited official submissions.

## Evidence and integrity

Follow the current official README and final repository README. Keep `src/api/`
compatible with the organizer's replacement. Compute the provided model for all
inputs, preserve evaluation behavior, and do not specialize to benchmark answers,
prompts, media, or performance flags. Do not reuse request computations across
requests. Record and reproduce embedded data from source and public tools.

Before a substantial experiment record the actual source commit, control,
complete execution change, expected removed/added work, correctness checks, and
the measurement that determines the next action. Reuse established evidence.
Preserve source variants, build provenance, failed runs and their actual causes.
Distinguish compile success, NPU correctness, measured speed, and official score.
Do not call a kernel test a full-model correctness test. Do not infer an E2E score
from kernel speedups. Keep working on concrete supported repairs, but stop
parameter sweeps without a specific remaining cost mechanism.

The user authorizes autonomous implementation, testing, official submissions,
private branch pushes, and final materials. Ask only for indispensable personal
facts or human authentication. Do not invent workshop attendance information.

- `media-vision-trf/`: batch_kernels exact B8-TRF vision projection with original K768 BF16 tree; isolated from media host integration.

- `tail256-model/`: batch_kernels model-only exact two-live-quarter P256 tail using existing P512 dense kernels; no Worker dispatch before whole-model gate.

- `tail256-worker/`: batch_kernels exact Tail256 Worker routing from a052254; retains real-owner512 priority, padded cost4, independent active allocations and per-layer decode responsiveness.

- `decode-r32/`: e2e_evidence bounded exact32-row hot-stage execution within the existing Rc64/Blocked64 host ABI; no serving activation before complete model/cache gates.

- `speculative-shadow/`: round2_path opt-in own-history n-gram shadow acceptance diagnostics from6f39574; no serving decision or model work changes, no NPU dispatch.

- `tail384-model/`: batch_kernels model-only exact three-real-quarter384 checkpoint from650642a; no Worker activation, original four-owner guard unchanged.

- `b16-ffn-epilogue/`: root exact post-attention, pre-FFN, native FFN and post-FFN launch composition; retains original HBM stage boundaries; no serving activation before complete gates.

- `three-owner512-model/`: batch_kernels model-only exact three-owner full512 constructor atop Tail384; private chronological quarter derivation, no Worker or NPU dispatch.

- `b16-prefix/`: round2_path isolated exact B16 RMS, quantization, blocked QKV and head/RoPE composition fromb078362; retained HBM stages, no serving activation or NPU dispatch.

- `packed-tail-worker/`: batch_kernels combined three-owner full512 and Tail384 Worker dispatch fromb078362, preserving c067 B16 epilogue and all real-owner priority/credit rules.

- `native-o-quality/`: root isolated approximate FP8 sliding O public-task gate from c067; preserves prior failed synthetic drift guard, never serving integration before actual quality/performance gates.

- `decode-r32-export/`: e2e_evidence clean tested32-capacity integration export on current B16-epilogue root; source-only handoff, no Arena.

- `b16-metadata/`: root isolated elimination of unused per-row legacy offset uploads when B16 uses pooled metadata; no model work or request reuse changes.

- `native-o-current/`: root approximate FP8 grouped sliding O composition on exact c070; retains q014 failed numerical evidence and requires fresh public/private quality gates.

- `b16-prefix-export/`: round2_path clean exact B16 prefix export onto6b55cb0 with R32/packed tails preserved; root owns NPU and serving freeze.

- `group-ffn-epilogue/`: round2_path isolated exact three-launch blocked Group/P512 norm-quant, current FFN core and post-FFN composition fromeb5004a; retained HBM boundaries, no cache/owner changes or serving activation before gates; reuses ffn-k256 target.

- `decode32-output-export/`: e2e_evidence clean exact32 O integration on6cd4878; validated kernel, scoped model gates, Automatic activation only after both whole-model gates.

- `b16-commit-heads/`: e2e_evidence isolated exact live-prefix B16 cache commit distribution from c070; same Rc64 pool and attention, no host dispatch before complete-chain gates.

- `ffn-globals-server/`: batch_kernels scoped default-off exact P512 FFN global-DM preload from c070 eb5004a, all-owner model/cache comparison; no Worker activation or Arena dispatch.

- `b16-attn-output-fused/`: e2e_evidence original live-prefix commit→B16 attention→H-split O retained-HBM composition;14/16 micro only before complete gates, no host activation.

- `headrows-local-sum/`: root isolated approximate local denominator reduction atop CURRENT exact P128 HeadRows; historical synthetic model drift failures retained, no serving change, reused root target.

- `full-o-native/`: root isolated approximate full-attention output FP8 activation projection fromq015, original eightK1024 BF16 tree and canonicaloutputs retained; micro first, no serving changes before gates. Reuses root target exclusively.

- `b16-attn-output-export/`: e2e_evidence clean exact original B16 commit/attention/Hsplit-O composition export on c070, Automatic activation after all-prefix and matched model/cache gates; Group/P512 flags untouched.

- `full-o-native-model/`: batch_kernels scoped default-off native full-O model drift diagnostic frome42a6bf; no serving activation or device build before parent micro gates.

- `group-ffn-export/`: round2_path clean exact global-preload plus grouped FFN epilogue export on c070 eb5004a; defaults stay off until separate joint all-owner gates, reuses ffn-k256 target; root owns serving/NPU.

- `native-o-composed/`: round2_path q016 approximate sliding native-O transplant from q0157425d5c onto exact e2aeb0b; ONLY kernel/export/sliding grouped call changes, no full-O alteration; reuses ffn-k256 target; root owns fresh quality/performance/Arena.

- `group-head-attn-commit/`: e2e_evidence exact four sequential native P128 head/attention/commit graph from e2aeb0b; one mutable pool pair, original HBM/math/order, micro only before complete-chain gates.

- `startup-3900/`: round2_path isolated q016 receipt-budget3900ms causal probe fromcc503bf; only budget and opt-in cohort release diagnostics change, original receipt/membership/credit/ownership preserved; root owns Arena and promotion.

- `ffn-projection-clusters/`: batch_kernels isolated exact P512 UP/GATE projection-owned clusters fromc071 f45b314; joined immutable weights, unchanged H960 math and original blocked outputs; complete FFN micro only, no serving/Arena. Reuses tail256-model target.

- `output-weight-stationary/`: root isolated original-BF16 sliding O weights in TRF with output-H cluster/slice ownership and main Switch input; K1024 BF16 partial/tree retained, compile/micro first; no serving activation. Reuses exclusive root r64-server target.

- `full-page-ingress/`: e2e_evidence exact immutable full-page K/V gather once plus Switch ingress experiment; unchanged QK/PV/page recurrence, no host activation, owned prefill target reused.

- `full-output-weight-stationary/`: e2e_evidence exact full O extension with two retained64KiB weight TRF epochs, original8K1024 BF16 tree and inclusive canonical output; micro only, reused prefill target.

- `output-weight-stationary-model/`: round2_path scoped default-off exact sliding O weight-stationary host integration on c07105f6f6f; all owner/cache model gates before activation; reuses ffn-k256 target, root owns Arena.

- `ffn-projection-clusters-model/`: batch_kernels scoped default-off exact projection-owned UP/GATE from c07105f6f6f; startup device join, composed same-norm graph, all-owner model/cache gates. Reuses tail256-model target; root owns NPU.

- `ffn-weight-stationary/`: root isolated full-H nativeFFN weight-TRF hypothesis, first4termF32reduction-order diagnostic; no model/serving activation; exclusive root r64-server buildtarget.

- `b16-l480-quant/`: e2e_evidence isolated exact B16 FFN L480-sharded hidden quantizer; original HBM/dense/norm paths retained, baseline stage trace and complete epilogue gate before host changes; reused prefill target.

- `output-weight-stationary-export/`: round2_path clean exact sliding/full O host export and separate authorized activation after all11 model gates; base c07105f6f6f, no Worker changes, reused ffn-k256 target.

- `ffn-projection-export/`: batch_kernels clean default-off projection-owned FFN transplant on current root91e5dd7, exact sliding/full stationary O enabled in both joint model arms. Reuses tail256-model target; root owns NPU.

- `ffn-up-weight-stationary/`: round2_path isolated exact L32/full-H UP projection with startup canonical FP8 copy and two TRF epochs; source3b76039, original H960 sequential F32 sum/scales retained, full old-output ABI conversion timed; no serving or Arena, reused ffn-k256 target.

- `qkv-weight-stationary/`: root isolated exact slidingQKV output-owned joined-DM/TRF experiment from91e5dd7; no model/serving change until complete gates.

- `b16-up-ingress/`: e2e_evidence isolated exact packetized B16 UP/GATE input-once→Switch→TRF; DOWN/quantizers/weights/math unchanged, complete epilogue gate before serving; reused prefill target.

- `b16-up-ingress-model/`: e2e_evidence default-off per-call exact UP activation ingress on c072624431f; unchanged R32/scalar/prompt routes, all-state/cache/sampling model gates before activation; reused prefill target.

- `b16-up-ingress-export/`: e2e_evidence clean exact shared UP ingress export on c073fd3c81d; separate Automatic activation after both actual model gates,111 tests/plaincheck; reused prefill target, root owns public/NPU.

- `qkv-weight-stationary-model/`: round2_path scoped default-off sliding grouped QKV weight-stationary integration on fd3c81d/c073; borrowed output views, all11 owner/cache oracle, no Worker/fullQK/O/FFN changes; reuses idle ffn-k256 target, root owns Arena/activation.

- `b16-down-permutation/`: e2e_evidence isolated F32 whole-slice L16/H16 DM permutation cost proof on c074; no DOWN math/host integration, reused prefill target, root owns Arena.

- `b16-attn-epoch/`: batch_kernels isolated exact B16 attention one-iteration scheduling epoch fromc074d810765; unchanged commit/QK/softmax/PV/O math and HBM boundaries, resource gate before micro, reused tail256-model target; root owns Arena.

- `qkv-weight-stationary-export/`: round2_path clean all11-validated sliding grouped QKV export onto a8b4d0f, preserving exact B16UP; scoped default-off import then separate authorized activation;111 CPU/check, reused ffn-k256 target, root owns Arena/server freeze.

- `recovery-cohort/`: e2e_evidence isolated default-off finite first-output recovery cohort froma8b4d0f; once-participated request state, existing-decoder/prior-prefill passthrough, unchanged3750/90/150 policies; CPU lifecycle gates, reused prefill target, root owns Arena/activation.

- `recovery-cohort-export/`: e2e_evidence clean recovery policy export onto c0750dc80b7; default-off import and separate authorized ON candidate, unchanged QKV/B16/model kernels,118 combined CPU gates; root builds/dispatches.

- `prefill-prefix512/`: round2_path isolated exact unwrapped P128 old-history prefix512 attention from0dc80b7; cropped K/V gather, original1024 score/sum and32-part BF16PV tree retained; resource-first then micro only, no host/default/Arena, reused ffn-k256 target.

- `ffn-trf-cache/`: batch_kernels isolated projection-owned P512 FFN P32 temporal-order/TRF-cache prototype fromabdcbb52; fixed Linner30 preserves H64/H960 order, current complete-FFN ABI and original control; resource-first, no host/default/Arena, reused tail256-model target.

- `b16-lm-partitioned/`: e2e_evidence isolated B16 BF16 LM H960-partial resource and strict all-vocabulary provided-weight micro from c0750dc80b7; original ABI and softcap boundaries, F32 association explicitly unproven; no host/Arena, reused prefill target.

- `b16-lm-storegroup-export/`: e2e_evidence clean exact B16 LM final-store grouping on recovery-OFF abdcbb52; one canonical-ABI call swap after25M actual-bit gate,118 CPU/check, no H960 arithmetic or policy changes; reused prefill target.

- `recovery-late-admission/`: e2e_evidence isolated default-off recovery cohort late-admission policy from c077e71cb50a; finite membership growth with unchanged original receipt deadline, meaningful lifecycle CPU gates, no model/kernel/Arena changes; reused prefill target.

- `norm-storegroup/`: batch_kernels isolated exact current projection-owned composed FFN norm qinput/xscale store grouping from4bab6ea3; same cluster and arithmetic/ABI, resource-first with inclusive DM costs, no host/Arena; reused tail256-model target.

- `norm-storegroup-model/`: batch_kernels default-off prepared-state exact grouped norm stores on current c077 math4bab6ea3; all11 real-owner/logit/KV gates before activation, no Worker policy; reused tail256-model target, root owns Arena.

- `int4-padded-trf/`: e2e_evidence isolated direct padded-i4 activation→TRF contraction legality/resource proof fromc077e71cb50a; no dense compaction, quantizer/host changes or Arena, reused prefill target.

- `norm-storegroup-export/`: batch_kernels clean norm-store grouping on c078ac2a8895; default-off kernels/scoped flags then separate activation only after all11 model gates,127 CPU tests, no Worker changes; reused tail256-model target.

- `norm-qkv-storegroup/`: e2e_evidence isolated current c079 grouped preattention normalize/quantize resource attribution; no production edits or arithmetic/host changes, reused prefill target, root owns Arena.

- `ffn-trf-l4/`: round2_path isolated single fixed Linner4/P32 projection-owned FFN resource gate from frozen97c610e7; same-tree P64/straightP32 controls, original H960 order and all boundaries retained; no serving/Arena, reuses idle ffn-k256 target.

- `ffn-projection-p32/`: round2_path one unchanged straight-P32 projection-owned FFN complete current c079 norm-store/residual micro from0ccbbf0c; all seven retained buffers/actual layers/feedback/ABBA, no L4/L30 or host activation; reuses ffn-k256 target, root alone dispatches Arena.

- `fp4-weight-stream/`: batch_kernels isolated B16 fixed-F4-to-E4 main-fetch weight compression feasibility fromc0796ac7dcff; dense-E4 identical-code control, resource/micro only, no model quantization or serving/Arena; reused tail256-model target.

- `b16-f4-reconstruct/`: e2e_evidence isolated exact provided-FP4/blockscale to current-E4 DM reconstruction at B16 weight load; unchanged native H960 dot/activations, resource bound first, no host/storage/NPU change; reused prefill target.

- `b16-lm-fp8/`: round2_path isolated approximate full-H B16 LM using a power-of-two-scaled E4 embedding shadow from394d60a2; original BF16 embedding retained, exact current storegroup control, capacity/resource first and no host activation/Arena; reuses ffn-k256 target.

- `int4-pack-words/`: batch_kernels isolated integer-domain signed-i4 nibble pack into dense owned-i32 words from394d60a2; no quantizer/model/host change, exact byte micro before any raw-ABI consumer integration; reused tail256-model target, root owns NPU.

- `int4-raw-binding/`: e2e_evidence standalone public checked Buffer binding of I32-packed bytes to normal compiled I4 consumer; integer/byte oracle, no SDK/Cargo/quantizer/model changes, reused prefill target; root owns Arena.

- `b16-lm-fp8-model/`: round2_path default-off B16-only approximate native LM shadow integration on current c079394d60a2; original embedding retained, real-history teacher-forced/memory/cache/selection gates before quality, no serving activation/Arena; reused ffn-k256 target after root archive coordination.

- `int4-power2-quant/`: batch_kernels explicit finite-BF16 power-of-two symmetric A4 quantizer from394d60a2; exact integer exponent/reciprocal and ties-away CPU/code proof, no old amax/7 relaxation or model activation; reused tail256-model target, root owns NPU.

- `b16-f4-group240/`: root approximate group240 F4 reconstruction resourceHOLD ec3b7173 (0.63percent gain), no NPU or host.

- `b16-f4-partialscale/`: root approximate F4 group960 directlookup dot and localpartial scale resource-first, r64-servertarget, no host.

- `b16-f4-partialscale-micro/`: e2e_evidence isolated complete B16 approximate per960 F4 weight-scale micro from8df08ec3; matching dense-code intended oracle, original-E4 error/performance baseline, no model/host dispatch; reused prefill target, root owns Arena.

- `f4-group-quant/`: batch_kernels pure CPU current-E4 to per960 power-of-two F4 quantization helper from8df08ec3; independent rational/layout proof, unchanged row scales, no activation/model/NPU changes.

- `physical-pool32/`: round2_path isolated exact32 physical sliding slots from c079394d60a2; R64 query batch, Fc256, context8192 and all model math retained, general FIFO admission32 with queued overflow; affine-address/resource/model/CPU gates before any promotion, no q018 merge or Arena; reuses ffn-k256 target.

- `p512-f4-partialscale/`: batch_kernels isolated approximate group960 F4 P512 projection-owned UP/GATE and DOWN with unchanged current norm-store/GeGLU/residual graph fromc079394d60a2; complete resource-first gate, no host/NPU, reused tail256-model target.

- `p512-f4-partialscale-micro/`: e2e_evidence complete current P512 norm-store/projection-owned approximate group960 F4 micro from2e7226c6; intended fixed-code dense oracle, all retained stages/feedback/permutation/F64 and currentE4 drift, no model/host activation; reused prefill target, root alone dispatches Arena.

- `physical-pool64-control/`: round2_path separately frozen exact c07964-slot control with identical bounded32-owner diagnostics only; no production storage/capacity change, shared ffn-k256 target used sequentially after physical32 build.

- `p128-native-qk-bound/`: root compiler screen of native E4 old-history QK in current fused P128 heads/attention/commit; direct casts deliberately unscaled, no numerical or serving claim, no Arena; r64-server target.

- `b16-f4-model/`: e2e_evidence default-OFF B16-only approximate per960 F4 shadow on physical32 07718f07; atomic streaming all48 startup, scoped Automatic-compatible diagnostics, unchanged E4/scalar/prompt/R32 and no q018; reused prefill target, root owns Arena.

- `b16-f4-model/`: e2e_evidence default-OFF B16-only F4 shadow/model diagnostics on frozen physical32 07718f07; atomic48-layer preparation with explicit borrowed per-call weights, unchanged scalar/prompt/R32, no q018; prefill target, rootArena.
- `p512-f4-partialscale-micro/`: e2e_evidence completed intended-code correctness but actual7.3–7.8percentgain below10percent gate; HOLD, nohost.

- `b16-f4-mse-micro/`: batch_kernels isolated B16 measured F4 partial-scale chain with deterministic MSE-fitted dyadic static scales fromae7b102f; same intended/reference kernels, complete provided-weight micro only, reused tail256 target, root owns Arena.

- `r32-f4-partialscale/`: batch_kernels resource-only current R32 direct F4/group960 partial scaling from physical32 07718f07; original activation/norm/store ABIs, no host/default/NPU; reuses tail256-model target.

- `physical-pool32-export/`: round2_path clean physical32 source-only export from c0794e024028; preserves organizer dependency files and original no-digest KV diagnostic, omits local SHA probe bins; functional equivalence audit, fresh server/check, reused ffn-k256 target. Root owns c081 freeze/Arena.

- `b16-f4-mse-model/`: e2e_evidence default-OFF B16 F4 preparation-time MSE enum and bounded model diagnostics; clean original Cargo/no ring digest; same measured kernel and physical32, no q018/P512/R32 F4; own prefill target, root Arena.

- `r32-f4-partialscale-micro/`: batch_kernels fresh MSE static F4 R32 intended-reference and full current three-launch FFN micro from98a6d7e5; no host activation/Arena, tail256 target.

## Shared target build provenance (c080 incident)

When reusing an agent-owned Cargo target across worktrees, refresh the package
source mtimes before a build. Cargo can accept relative dep-info from another
worktree when returning to older source files. Use root `tools/record_run.py`
with `--fresh-package --capture-artifact target/release/server` for server builds;
other binaries may use their actual relative artifact path. The target must
remain exclusive during the build and capture. The wrapper verifies unchanged
source/config hashes after a successful build and freezes its output immediately.
Use that captured output for candidate freezes, never a later mutable target.
A compile success or source hash list alone does not prove library freshness.
Preserve historical artifacts; c080's d40 runtime contradicts its recorded Rc32
source and is quarantined. New physical32 servers must verify actual allocated
buffer extents and admission capacity at startup before quality/performance claims.

- `b16-fixed-e4-kv/`: root fixed-unscaled E4 sliding KV codec plus current B16 commit/attention/O resource bound; no pertoken scales, original BF16 QK/PV retained, approximation and public legality unproven, no host; exclusive r64-server target.

- `b16-f4-mse-quality/`: e2e_evidence isolated public-quality-only MSE F4 B16 candidate from 2d0bb5a8; explicit immutable MseDyadic startup and only existing independent canonical B16 Automatic scope, source-only original Cargo; no root integration or Arena by agent.

- `f4-mse-parallel/`: round2_path optional deterministic CPU-only parallel per960 MSE encoding from q01913174d8b; same serial arithmetic and ordered bytes/scales, capped standard scoped threads, no startup/default/D2H changes or Arena. Own ffn-k256 target if needed.

- `external-quality/`: round2_path independent preregistered GSM8K TRAIN32 diagnostic harness from933276c4; seed20261026, two fixed16-row shards, organizer prompt/scorer unchanged, frozen c081/q020/q021 servers only. No serving source edits/builds or Arena dispatch; root owns execution.

- `b16-fixed-e4-kv-micro/`: batch_kernels isolated fixed-unscaled E4 KV complete B16 append/attention/O oracle from27471cbf; live14/16, actual O weights and explicitly synthetic cache history; no host/Arena, exclusive tail256 target.

- `fixed-e4-kv-quality-model/`: e2e_evidence default-OFF fixed scale1 E4 sliding KV numerical surrogate on clean physical32c0813d497087; all10 producer hooks, BF16 storage and unchanged readers; no compressed port or Arena by agent.

- `p512-f4-predecode/`: root isolated one-time F4→E4 DM expansion reused across current P512 query chunks; retains group960 partial scales, old inline7.6percentHOLD remains, complete resource gate/no host; root target.

- `b16-f4-mse-lm-fp8/`: batch_kernels default-off q019 MSE B16 FFN plus q018 B16 FP8 LM composition from13174d8b; same-FFN teacherforced/cache/replay gate, no parallel startup/Worker/default/Arena; exclusive tail256 target.

- `fixed-e4-kv-quality-on/`: e2e_evidence separate fixed-scale E4 KV ON numerical artifact; only constant differs from paired OFF source, original BF16 pools/model otherwise retained; reused prefill target, root alone dispatches.

## Device-kernel cache provenance (fixed E4 control incident)

The SDK 0.9 shared device-artifact cache reused ON kernels after an external
Rust device-graph constant changed to OFF, even with fresh package source mtimes.
The affected compiled bins differ after a clean rebuild while their stored
fingerprint files remain identical. Source hashes and host flags alone therefore
do not establish the intended device graph.

For a device-graph constant change, use a new empty `FURIOSA_OPT_OUT_DIR` through
the documented build option, then capture the resulting executable immediately.
Verify changed caller images and unchanged controls, and run the original-path
negative control before interpreting an A/B comparison. The old a006 fixed-E4
OFF artifact and its apparent zero-drift comparisons remain quarantined; preserve
the failed head/scalar controls. A pure host switch among already verified device
functions can use an explicit registry-image parity audit instead.

Parse captured ELF sections read-only. Do not run `objcopy` on an original
captured ELF: even `--dump-section` can rewrite its input when no output ELF is
specified. Preserve audit incidents and verify frozen hashes after inspections.
