# Checkpoint during provisional submission coordination hold

Functional source: `340919dd329cf25516544e99711fc6048f5d0e2c`. No new work is launched pending the root's scope decision. All existing runs are terminal; own prefill target is idle. No root source, Worker/default activation, Arena job, or official candidate changed.

Implemented, not yet hardware validated:

- Explicit `Workspace::new_canonical_fixed_e4` and typed `SlidingFixedE4` pools. Ordinary constructor stays BF16. Scalar, B16, R32/R64 and every real prompt state dispatch to the corresponding new pool dtype. Full attention, masks, current q022 head rounding, admission32 and query64 remain unchanged.
- Sliding metadata uses one byte per element; RoPE/full/logit metadata keeps BF16 bytes. Model construction fixes preparation metadata width; all 79 async Model entries accepting a Workspace now reject another dtype before device work. This closes the peer-found possibility of supplying an E4 Workspace to a Model originally created with BF16 metadata.
- Direct scalar gather/LUT and token scatter, so the current source allocates **no additional persistent ring scratch**. The 5 GiB pool difference is arithmetic only until owning-buffer/allocator evidence is collected.
- Sequential exact probes `compare_fixed_e4_storage` and `compare_fixed_e4_storage_packed`. They retain the established same-storage replay, then compare every recorded full-vocabulary row across storage. A 12 GiB host snapshot covers all cells of all32 sliding slots and all256 full pages; candidate E4 bytes are independently decoded on CPU for exact comparison. One device model/pool exists at a time; immutable weights survive a consuming Model boundary. This has typechecked but not run, so time/memory limits remain unverified.

Completed checks:

- a001 initial scalar utility typecheck failed because a compound generic row axis nested differently from flat local axes. `a7d47d5c` fixes it with explicitly flat head-row helpers; the failure is preserved.
- a002 combined host/server typecheck passed in 12.74 s at `7acf16c0`.
- a003 direct scalar attention and a004 scalar QKV/commit resource compiles passed, both without `DramStack` in their emitted resource IR. This is not runtime correctness or speed evidence.
- a005 both new model probes plus plain server typecheck passed in 11.43 s at `340919dd`.
- batch's bounded 7acf16c0 host audit found no decode byte-offset/live-prefix/policy-order defect. round2's prompt audit found no prompt ordering defect and identified the Model/workspace dtype pairing gap fixed in 340919dd; that guard correction has not yet received a final peer audit.

Imported peer device sources: prompt34444109 + type repair199bc39b, decodee2d021f2. Original arithmetic modules remain intact; import conflicts were additive lib exports only. Peer resources/micros and their outstanding states are recorded in their separate worktrees. No compressed actual-model/public/HTTP result exists.

Still required before any activation: execute meaningful CPU tests including the two newly added guards; device-build and run all-live-prefix/reader/prompt/scalar/idempotence gates; fresh-capture model binaries with image binding; exact sequential cache/logit gates including all11 packed owner layouts and active P128/scalar interleaves; actual owning-buffer/memory proof; then root-controlled quality/performance evaluation. Source is deliberately incomplete as a validated candidate and must not be submitted as the existing q020 or q022.
