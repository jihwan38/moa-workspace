# Clean exact32 O integration

Base: root6cd4878 (c069 and compact B16 metadata). No source conflicts. The new
kernel is byte-identical to actual a011/b45a6f2. The scoped host was tested in
actual a013/662e309; this export additionally selects it for Automatic after both
full-model gates passed. Root performs the combined plain server build.

Functional files:
- `src/ops_decode32_output.rs`: exact first32 sliding/full O and partial diagnostic
  functions, retained64 HBM and original K1024 BF16 trees.
- `src/lib.rs`: one module export.
- `src/host/decode_r64.rs`: one per-call boolean gated by existing capacity32;
  only the two O dispatches change, with no new scratch or weights.
- `src/host/model.rs`: scoped output diagnostic/control and Automatic activation;
  public validation, cache ownership and all other selectors preserved.
- `src/bin/compare_decode32_output.rs`: actual paired-history/all-logit/cache
  oracle, active interleaves, all-width guards and stage timing.

The experimental synthetic/supplied-weight micro helper is retained in the
original decode-r32 tree, not copied into this minimal integration export.
Source-only selection needs no Cargo features. No src/api or Worker changes.

Actual correctness/performance provenance is in `decode32-output-actual.md` and
`decode32-output-model-actual.md`. Prior stage profile explains the selected cost.
Plain integrated checks and the unfiltered111-test suite are recorded separately
when complete; older isolated88-test results are not described as111 tests.

## Integrated checks

Functional commit9652aec, a015/a016 records:
- Plain `cargo check --bin server --bin compare_decode32_output` passed.
- The ordinary unfiltered test build in a015 reached the linker, then failed
  because combined DWARF debug sections exceeded32-bit relocation range. No
  tests ran; this was not a test assertion or source compilation failure.
- a016 reran the identical source and full test selection with
  `CARGO_INCREMENTAL=0 cargo test --lib --config profile.test.package.furiosa-opt-gemma4.debug=0`.
  Only this package's test debug emission changed; existing dependencies were
  reused. All111 tests passed,0 failed/ignored/filtered (0.20s execution,92s build).

The failed a015 record and generated cache are preserved. Root's archival lane
owns any lossless cache archival; no source, evidence or frozen ELF was deleted.
No further device build is needed from this export: root compiles the combined
candidate using plain production flags after cherry-picking9652aec.
