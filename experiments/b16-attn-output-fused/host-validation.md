# Scoped model gate preparation

Functional source: `0bf8c53`. `Dispatch::B16AttentionOutputDiagnostic` is an explicit per-call mode; Automatic stays unchanged. Every eligible canonical B16 call uses the original `PooledStep` offsets and masks and writes the same Prefill attention/projected tensors. The candidate differs only at commit→attention→Hsplit O. R32/O32 and R64 dispatch, prefix fusion, compact metadata, FFN epilogue, retained logits, and device selection policies remain matched.

The independent batch agent reviewed selector scope, physical owner allocation, live-prefix matching, retained output lifetimes, active P128 scratch separation, and original-O fallback/reentry. No source-level blocker was found. Width1 deliberately remains scalar; the fused1 device graph is covered by the all-prefix micro, not the serving host comparison.

Actual checks:

- `runs/a005-b16-attn-output-host-check`: cargo check PASS9.74s.
- `runs/a006-b16-attn-output-host-tests`: unfiltered library suite PASS111/111, 0failed/ignored/filtered;1m36s build,0.25s tests. CARGO_INCREMENTAL=0 and package test debug0 prevent the earlier unrelated large-DWARF failure.

The model probe independently computes17 real512- or1024-token histories for each side, with disjoint physical slots/pages and identical native arithmetic. It compares90 complete decode vocabularies plus the active prompt output, samples greedily and stochastically, exercises retained-output fallback/all-banned policies, and scans all18 matched cache owners. Widths1/2/8/14/16/17, reset, page/ring transitions,48 active-P128 quanta, original-O/canonical-QKV fallback,31 narrow prompt rows, and fusion reentry are represented. Actual NPU model results are pending; no model/HTTP speed or correctness claim follows from host checks alone.
