# SDK 0.9 cache and q021 artifact audit

Read-only audit, 2026-10-25 UTC. No build, Arena launch, SDK edit, source edit, cache deletion, or candidate mutation. q021 tree remains clean. Existing ELF files were read; only /tmp reports were written.

## Verified public runtime contract

Installed SDK root: /home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f (inside moa2026-round2).

- furiosa-opt-macro-0.9.0/src/lib.rs:330–354 constructs the launch selector from the fully qualified function path and declared numeric generic arguments: AxisName type arguments contribute SIZE, const parameters contribute their value, lifetimes contribute nothing.
- furiosa-opt-std-0.9.0/src/backend/npu/registry.rs:1–26 documents the embedded furiosa_kernels wire format as bincode Vec<(function path, Vec<u64> key, image bytes)>. Concrete functions have an EMPTY key. Lines32–47 deserialize once and filter by path.
- backend/npu/function.rs:228–239 matches that path/key to an embedded image. backend/npu/backend.rs:44–57 caches the loaded image handle by (path,key) within that opened device instance. This is not the build cache; it cannot explain an already-identical pair of executable kernel sections.
- furiosa-mapping-macro-0.9.0/src/lib.rs:101–108 expands axes declarations into a named struct and an AxisName implementation with associated NAME and SIZE. parser/mod.rs:215–228 places that size into the mapping expression.
- cargo furiosa-opt --help publicly documents FURIOSA_OPT_OUT_DIR, default <workspace target>/furiosa-opt/kernel. Therefore an empty unique output directory is a supported way to avoid reuse of existing kernel-image cache files without deleting historical artifacts.

## Compiler-cache limit and observed failure

The installed compiler driver is a stripped prebuilt executable; npu-opt/src/bin/furiosa-opt-driver/cache.rs source was not available in the installed source trees. Binary strings mention Fingerprint and Dep::Body/Attr/Adt/Impls, but this does NOT establish their exact hashing algorithm, dependency traversal, or treatment of associated/free constants. Binary .hash files were not decoded. Do not describe the runtime path/key as the complete compile-cache key.

E2e/root preserved direct evidence for the fixed-E4 failure: only the external device helper ENABLED constant changed; the fresh host rebuild emitted no device Compiling lines; shared .bin/.hash mtimes remained from ON; three corresponding ON/OFF ELF kernel sections were identical; and OFF negative head/scalar controls failed (r1324/r1325). This establishes stale device-image reuse across this external-constant edit. The precise internal dependency omission remains unproven without driver source. A fresh package/source hash capture alone does not prove device-cache invalidation.

## q021 delta is a different, verified path

Captured default-OFF diagnostic ELF:
/round2/runs/a003-b16-f4-mse-lm-fp8-model-build/built/target/release/compare_b16_f4_mse_lm_fp8
SHA256 15ff8c81dc6db22a07069ec3790ad3b3f1d7cf79a5be13375d0cd18c08afa2c4

Captured activated server (source039b440e):
/round2/runs/a006-q021-mse-lm-fp8-server-build/built/target/release/server
SHA256 7d89bb85e53fd9cd1cfcc44ebeaed4226aafe5f9cdb2063ac270ddcf4a272072

Decoded public bincode registry: model306 entries, server305 entries; ALL305 common (path,key) entries have identical image bytes. The extra model entry is diagnostic-only. Complete per-entry hashes are in /tmp/q021-kernel-registry-audit.json.

Distinct concrete entries (empty generic keys):
- ops_b16_lm_fp8::logits_quantized: 56,832B, SHA2272562f243038e2ef67f6dfdae1b52bdf39ebf4019a9122544f9cb02d39531e.
- ops_b16_lm_storegroup::logits_batch: 51,968B, SHA6c3bfc44d177da6ffac7c66e6471f52926d661387976d6b5d4bb1e4c2a186996.
- ops_b16_lm_fp8::quantize_embedding: 50,432B, SHA05af86e8825e8c69519a182a73b3f6b795b12e24ee1b3389de5cfccc136168b8.
- ops_query_b16_native_fp8::quantize_input: 45,056B, SHA5355201189982d02cd8dbe9acee743bd8ce799c0d885549ef553fa282f5bd1c9.

B16_LM_FP8_ENABLED is used in HOST Model startup/dispatch. Prefill::decode_batch chooses distinct already-compiled functions. Both explicit model controls exercised those named functions before activation; activation does not ask an unchanged device function to reevaluate an external constant. Source ops_b16_lm_fp8.rs remains byte-identical to q018 c79e7f30 (SHA38cd8a24fe153e90d625b4450358b98b0bf470a7d588d7358daf775134035390).

The two q020 imports match d52f0c08 byte-for-byte: host/f4_group960_mse.rs SHA6bfa34bd9c379b846cc37424e09373afbaf6095671949a3c8390df5e6389949a and host/b16_f4_weights.rs SHAaf0a03d3dbbb8b7d96aee1fa7d9ba8fddc4e45e2a555f53f6adfd651835ebf5c. They change CPU weight preparation, not a device-constant decision. Conclusion is scoped to these q021 deltas and these captured images; not a blanket historical kernel-validity claim.

## Physical32: inspect actual artifacts, not assumed key sufficiency

Old source89c0097c declares Rc64=64; compact source07718f07 declares a distinct Rc=32 and compatibility alias Rc64. This changes type identity/associated size and mappings. However concrete pool functions still use EMPTY runtime keys: the public runtime key alone does not prove automatic compile-cache invalidation.

Independently decoded preserved physical32/64 artifacts:
- test_physical_pool:21 common entries,19 different images.
- compare_physical_pool_history:301 common entries,50 different images.
- compare_physical_pool_group:301 common entries,50 different images.

Changed images include zero_sliding_pool, all16 composed B16 attention/output variants, ordinary B16 sliding attention and both fused P128 head/attention/commit entries. zero_full_pool stays byte-identical. Per-entry details: /tmp/physical-pool-kernel-registry-audit.json.

Existing compact resource JSON explicitly declares134,217,728B sliding-pool allocation and a32-iteration zeroing loop. Preserved cross-program evidence reports41 matched memory observations differing by exactly10,737,418,240B, complete physical raw-address gates, and775 equal ordered model/cache digest events across15 paired runs. These are positive evidence for this actual physical32 change, beyond a source-only key argument. No similar confirmation is claimed for unrelated historical constant-only edits or uninspected artifacts.
