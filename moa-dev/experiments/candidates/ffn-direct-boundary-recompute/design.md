# FFN fixed-shape first-row recomputation: preparation only

Prepared 2026-09-27; execution worktree: /home/jihwan38/moa-worktrees/ffn-boundary.
Root authorized candidate-directory implementation and diagnostic compile/Arena execution on opt/ffn-direct-boundary-recompute, starting commit0a4417c076eff78ba885b0f142c66c81e9bc9b4f. Live crate and baseline remain unchanged; no commit/push before root review.

## Scope, origin and changed mechanism

This is a conditional repair of the existing direct-FP4/native up/gate execution family, not a new independent architecture. Intended optimization is eliminating full up/gate FP8 DM materialization while retaining a small, explicit materialized exception at a fixed layout boundary. It is not an input-dependent/fixture-dependent path.

Arithmetic/source origin is baseline E104, commit 7f891f9c48e670fe014b2ac5bd2c92bae3bac433. Direct producer origin is the frozen fp4-native-boundary/v7c/probe.rs. Actual branch starting commit must be filled by root after QKV/O record integration; it is distinct from the source recipe and performance control.

Relevant frozen files:

- moa-dev/experiments/candidates/fp4-native-boundary/v7c/{probe.rs,boundary_probe.rs,mlp.rs,layout.rs,lib.rs,recipe.json}
- furiosa-opt-gemma4-12B/src/device/shared/mlp.rs (E104 project_rows_with_scale_contraction, project_up_and_gate and input quantizer)
- moa-dev/experiments/candidates/fp4-native-boundary/{README.md,build.py,build-boundary-v7c.json,boundary-v7c-results.json}
- baseline/e104.json and docs/research/top10-audit-20260927.md

Prior failures: C07 direct native projections; v6 two independent direct streams corrupt nearly all nonzero rows; v7c joint producer leaves projection0 row0 on slices0..7 in both clusters wrong, projection1 passes the four checked input cases; v8 restoration-scale movement, v9 row padding lowering, and v10 duplicate-projection warmup did not repair it. Later materialized debug-output controls timed out. M04 retained every FP8 materialization byte and regressed; M05/M07 changed preparation/ingress and did not improve M03.

New mechanism: keep v7c's entire joint producer shape and calculation, then recompute projection0 local row0 on ALL 512 slices using a compact materialized FP8 path. Replace only that fixed row in the direct result. Do not reduce the direct producer to 29 rows, restrict the repair to the observed 16 bad rows, add dummy warmup, or grow fallback coverage after failures without a new diagnosis.

## Preferred merge boundary: F32 block partials

The proposed report described replacing final F32 rows. A more direct implementation replaces the row's F32 block partials before the unchanged weight-scale/reduction pass. A row has 240 F32 block partials = 960 bytes, already a multiple of the 8-byte commit alignment. This avoids a 4-byte final-row overwrite and the associated padded two-value read/modify/write. It also reuses the original full block-scale load and six-row reduction exactly.

The fixed recomputation region and the tested question are unchanged: projection0, local row0, all slices and both clusters. Final scalar rows still use the existing six-row -> two-F32/8-byte commit path. The fallback's new one-row compact source/destination shape is not presumed hardware-correct merely because E104 materialized arithmetic passes.

Diagnostic graph:

BF16 H3840 HBM
  -> identical E104 runtime quantizer in DM (amax, global power-of-two scale, 3 E5M2 terms)
  -> ring32 distribution to 2 clusters x 256 slices; retain term DM and scale DM
  -> original joint TRF/VRF loads

FP4 up HBM + FP4 gate HBM
  -> original two transfers into joint FP4 DM [Projection2,Row30,H3840]
  -> unchanged Main lookup E5M2 + native DPE + input-scale restoration
  -> F32 partial DM [Projection2,Row30,Block240]

FP4 up HBM view [Cluster2,Slice256,Row30,H3840], Row=0 only
  -> compact FP4 DM [Row1,H3840]
  -> Main FP4->E5M2 lookup -> compact E5M2 DM [Row1,H3840]
retained quantized activation DM + input-scale DM
  -> separate fallback TRF/VRF reload, reflecting the validated E104 reload boundary
  -> native DPE with identical packet64/block16/three-term order + input scale
  -> compact F32 partial DM [Row1,Block240]
  -> supported typed transfer or explicit Main copy into joint partial [Projection0,Row0,Block240]

repaired joint partial
  -> unchanged v7c per-projection full scale DMA, five six-row scale/reduction passes
  -> complete F32 up and gate rows -> diagnostic F32 HBM outputs
  -> host independent oracle checks ALL 2x15360 rows for each input

The joint partial buffer is mutable only at the fixed repair. Its original full producer is retained. Compiler memory dependencies must establish original joint write -> repair write -> reduction reads; Rust lexical order alone is not proof. Dump graph/schedule and verify the emitted dependency and actual transfer extent.

## Placement, ownership, lifetimes and costs

Shared row mapping remains l=7680*c+30*s+r; c=0,1 and s=0..255. Original joint local row/time structure, FP4 code order, term=3, packet=64 and block=16 do not change.

| Intermediate | Placement/owner | Lifetime/consumer |
|---|---|---|
| quantized x, 3x3840 E5M2 | DM, all 512 slices | joint and fallback TRF reloads; lifetime may grow |
| input scale F32 | DM then separate VRF loads, all slices | joint/fallback dot restoration |
| joint FP4 [2,30,3840] | DM, 115200B/slice | one unchanged Main direct producer |
| joint F32 partial [2,30,240] | DM, 57600B/slice | fixed repair then unchanged scale/reduction |
| fallback FP4 [1,3840] | DM, 1920B/slice | lookup; source is row0 every 30 HBM rows |
| fallback E5M2 [1,3840] | DM, 3840B/slice | separate Main native dot |
| fallback input | TRF, 11520B/slice; scale VRF padded32B | fallback native dot |
| fallback partial [1,240] | DM, 960B/slice | 960B overwrite of row0 in joint partial |
| final up/gate | DM F32 30 rows/slice | existing 8B commits and full F32 HBM diagnostic output |

Versus direct v7c, logical added FP4 HBM read is 512*1920 = 0.9375MiB. Added FP8 materialization write is 512*3840 = 1.875MiB. Added compact F32 partial write is 512*960 = 0.46875MiB. A copy into joint partial adds another 0.46875MiB write/read; direct commit_view may eliminate that copy only if supported and dependency-safe. Additional TRF/VRF reload, lookup setup, native contraction, DMA commands, retained activation lifetimes and source-strided HBM traffic must all be counted.

Compared with E104, up/gate full FP8 materialization write totals112.5MiB; the direct main producer removes it but forces BOTH full packed matrices ready before one joint dot and holds both full partial sets together. The small fallback number is not a predicted speedup. Down retains E104's56.25MiB FP8 materialization. No source weights or inputs are transformed outside the timed submitted kernel.

For the diagnostic only, two final F32 vectors are written to HBM; this is an added boundary and invalidates diagnostic timing as whole-FFN performance evidence.

## Numerical contract and oracle

Retain finite BF16 input, actual E104 3-term preparation, input-scale restore before weight block-scale, native block16 F32 partials and F32 block reduction. No BF16 rounding before final projected-row boundary. No arithmetic masking that consumes NaN/Inf in replaced rows: overwrite memory, do not multiply bad values by zero.

First gate uses the v7c host's four precisely representable BF16 cases (varying signed small values, cancellation, all-zero, large magnitudes) and independently generated FP4 weight codes/scale per row/block. Every F64 oracle block and final row is asserted representable in F32; require exact comparison, all finite. For each case check both projections and all15360 rows, including all512 fallback rows and every untouched row. Log mismatches by projection, cluster, slice and local row; summarize repaired vs untouched coverage separately without relaxing overall success.

For paired control, the old shared-TRF materialized debug image timed out and cannot silently be called a passing control. Adapt the E104 materialized arithmetic with its per-projection activation/scale reloads to the same full F32 diagnostic output boundary, or pair with an immutable known-passing original C07 materialized oracle. Record exactly which comparison is passing. Candidate independent-oracle success remains necessary even if a different control passes.

If the first-row materialized path or typed transfer fails compilation, inspect precise source/transfer/consumer shape and support a concrete axis/alignment repair. A raw Main copy of compact960B partials is a supported-route alternative to typed-DMA expansion, analogous to M07's understood source repair. Do not substitute a changed numerical boundary or enlarged fallback just to get a build.

If untouched rows/projection1 become corrupt, or a correctly mapped fallback is wrong/nonfinite, freeze the evidence and close this fixed-boundary hypothesis; do not grow fallback coverage. If control times out, identify it as unresolved; do not claim all-passing paired validation. No unconditional compile-count abandonment for a diagnosed source bug, and no padding/setup sweep.

## Whole-FFN path after first gate only

Integrate the passing direct+repair projection into an E104 source recipe first, preserving E104 GeGLU, down, pre/post RMS, BF16/global-scale boundaries, residual and layer scalar. Return F32 rows in canonical L/60 layout and cast exactly where E104 does. M03 local GeGLU/down is a later compatible composition choice, not part of the first gate.

Original public decoder_feedforward signature, BF16 input/output, FP4 weights, per16 FP8 blockscales, F32 global scales and harness stay unchanged. Whole FFN must include every fallback load/transform/reload/merge. Run contemporary E104 -> candidate -> E104 in one allocation; QKV/O images must match control. Ordinary fixture tolerance remains atol0.01+rtol0.01*abs(reference) for FFN. Only a correct promising complete path proceeds to repeated NPU and expanded inputs. Keep static cycles, isolated diagnostic timings and full-kernel cycles separate. No official submission or adoption is implied by preparation.

## Tooling adaptation and files needed

Existing fp4-native-boundary/build.py constructs an isolated crate from recipe.base_commit plus pinned lock, verifies container file hashes, freezes per-attempt source/logs, builds three diagnostic images and verifies each image is embedded exactly once. Its HERE/ARTIFACTS/remote paths and hard-coded decode/materialized/fused modes require adaptation for a NEW candidate directory, not modification in the historical candidate. It uses a shared /tmp/moa-c01-probe/target and cargo clean; root must serialize compile use with other agents.

The first candidate directory needs README.md/recipe.json, frozen {probe.rs,boundary_probe.rs,mlp.rs,layout.rs,lib.rs}, isolated build.py, read-only verification/collector support and source/build records. Existing decode image is unrelated to the new question; preferably retain an unchanged fused replay and a materialized reload control plus the hybrid image. The host should select modes explicitly and use the unchanged independent input/oracle generation. Keep the immutable old v7c fused image replay available if the source's added unused kernels affect build layout.

Root will provide candidate ID/branch and source-write permission. Implementation will touch only that candidate directory; live crate, baseline, shared registry, Git, builds and Arena remain root-controlled until separately delegated. Before first build preserve actual branch starting commit, recipe origins/hashes, full graph and first stop condition, and run the appropriate source-plan check without implicitly selecting the adopted source as origin.
