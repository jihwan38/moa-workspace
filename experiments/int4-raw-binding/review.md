# Frozen checked raw-I4 review

round2_path independently reviewed 3145fb3b: exact whole-owner byte extents,
256-byte alignment/overflow, three argument slots and output disjointness match
the SDK public binding contract. Function submission retains Buffer clones;
Launch Drop drains completion before release. The immutable I32 owner and
exclusive output borrow persist through await. No dtype alias or private mutation.

The independent byte/word encoders, I64 dot oracle, all signed nibble cases,
true paired permutation, typed/raw/joint checks, rejected-short no-write check,
owner reuse and inclusive joint timings were found coherent. This is source
review, not device validation or cancellation stress evidence.

Later repairs 3c1e58bc/a02a95ff only import PackNibbles locally for the mapping
macro and make shift fixture Vec inference explicit. The consumer, adapter and
imported pack producer are unchanged. a003/a005 host failures are retained.

Frozen a007 release build succeeded in 34.14 seconds and includes both ordinary
typed kernel images plus the NPU-only checked adapter. All build source hashes
match. Hardware execution remains root-owned and pending.
