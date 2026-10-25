# Skip unused legacy row-offset uploads in pooled B16 decode

Control source: 6b55cb0, exact c067 epilogue plus R32 and packed prompt tails.
`Prefill::decode_batch` constructs a `PooledStep` for independent canonical
requests, then uploads three legacy scalar offsets for every row anyway.
`decode_sliding` and `decode_full` consume those offsets only in their unpooled
fallback branches. Logit extraction has a separate per-row offset and remains
required when the caller asks for materialized rows. Soft-token embedding
insertion has its own offset and also remains unchanged.

Change: keep legacy offsets in an optional typed structure, create them only
when no `PooledStep` is available, and retain the existing per-request values in
every unpooled fallback. A diagnostic switch retains the old allocation path
for matched comparisons; serving stays disabled until the model gate passes.
This removes three small PDMA uploads/allocations per live pooled B16 row,
up to 48 per decode, and changes no kernels, math, KV layout, token work,
prompt scheduling, or cache ownership. No metadata is cached across calls.

Gate: compare enabled/disabled full-vocabulary logits, deterministic and
stochastic selections, all allocated KV bytes, positions/page/ring reuse,
widths 1/2/8/10/16/17 and prompt interleaves. Exercise legacy fallback
separately, preserving its offsets. Record completed `prepare.decode` and
whole-model timings and then ordinary HTTP accuracy/performance. This is
a bounded host cost removal, not a claim of a major leaderboard gain.
