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
