# Default-off grouped norm-store model gate

Base is current c077 math4bab6ea3, with passed micro kernel44dfb082 and
r1151 exact393,248,768-byte/7–8% complete-chain evidence. No Worker change.

The selector belongs to each real PreparedGroup/Tail256/Tail384 and their owner
wrappers. It defaultsfalse, rejects selection after any real state starts, and
requires current projection-owned FFN. All legacy disabling controls clear it.
The common execution copy preserves it only through its prepared state. Only the
sliding/full FFN composed-call producer changes; attention, norms/math, residual
ownership, cursors and output mappings remain unchanged.

The existing11-route oracle keeps projection-owned FFN, preload/composition,
stationary sliding/full O and stationary sliding QKV explicitlytrue in both arms.
Only grouped norm stores are false/true. Both arms independently compute identical
true history on disjoint allocations. Candidate-first reset with NaN dense-only
padding, all-vocabulary outputs, all48-layer raw KV and paired sentinels, and
current B16/R32 interleaves are retained. Run onecase0..10 per invocation; optional
secondargument1 enables interleaves (defaultcase0). No threshold or prior control
is relaxed. Initial prepare/layer/finish timing excludes downloads/cache scans
and candidate-only decoder interleaves; stage aggregates include interleaves.
