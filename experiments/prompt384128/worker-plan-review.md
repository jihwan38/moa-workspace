# Worker integration after the exact two-owner model gate

Source dependency: passed constructor07fcb9e (root integrated asce88461).
The global PROMPT384128_ENABLED candidate applies to all eligible inputs.
Four oldest eligible independent prompts retain first priority. Otherwise, if
the oldest two prompts are128-aligned and page-local P128 eligible and the first
has384 available inputs, they retain exactly the A384/B128 ranges. Older short
or unaligned tails are not bypassed. Singleton Prompt512 fallback is unchanged.

Generation::prompt_chunk_count adds only an explicit limit==384 opt-in, with
alignment, availability and checked-end guards. Existing128/B16 and512 limits
retain their behavior, including512-limit short-tail fallback to128. Borrowed
inputs, retained ScheduledInput ranges, preflight validation and final progress
all agree on384/128; Generation never advances at an intermediate layer.

The pair uses the existing512-row cadence cost class and the same receipt,
credit, admission and reservation budgets. Both Model rings resolve active
ranges after session-vector compaction. The Worker routes the pair through its
separate start/advance API and uses the existing owner-ordered output sampling.
complete_work, reservation release, terminal handling and offline cleanup are
unchanged. No server was built here; root integrates into the current candidate.

All73 host tests passed. New tests exercise quad priority, singleton512 fallback,
short/unaligned turns, exact two-owner range retention after decoder removal,
None/short-output rejection before mutation, cancelled-owner reservation holding,
and nonfinalA progress without sampling. round2_path independently audited the
selector, ownership, cancellation/error and decoder-interleave paths and found
no concrete issue. HTTP/SLO evidence is still required before a serving claim.
