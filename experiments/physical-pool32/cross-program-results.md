# Physical64 versus compact32: matched complete results

Root's original64 r1229..r1244 queue completed all16 jobs. Sources are old64
89c0097c and compact32 07718f07; they use the same logical32-owner schedules and
same current c079 arithmetic. The only diagnostic text change is the final Group
physical64 label. All original64 in-process paired-owner/cache gates passed.

The15 model log pairs match all775 ordered SHA256 events, covering169,082,880
individually reported logit values and48,242,884,608 reserved-cache bytes per
program. This is cross-program SHA256 equality, in addition to each program's
independent in-process paired-owner bit/byte comparison. It is not described as
an in-process64-versus32 full-byte comparison. The775 event comparisons also match
owner labels, positions, physical ring IDs, page counts and byte/value counts.
Every log must contain its final PASS; incomplete logs are rejected.

The physical micro has71 zero-difference comparisons in each program.67 labels
have identical extents: all32 live ring readbacks, immutable source rows and the
unchanged full-page pool. Four full sliding-pool checks have the declared2x old64
extent; both independently match the same affine CPU address/value model, with
old slots32..63 untouched and zero. Totals are738,263,040 BF16 values in old64 and
469,827,584 in compact32. These different-sized arrays are not counted as a direct
same-size raw comparison.

All41 matched memory snapshots show exactly10,737,418,240 bytes (10GiB) more
available memory in compact32. This is measured allocator availability, matching
the predeclared40layers*2pools*32rings*4MiB payload reduction. Full pages, context,
query capacity and model math remain unchanged. Warm compact availability is
approximately14.276GB with existing joined FFN weights and lazy scratch present.

The fresh plain-server build a005 captures the resolved Cargo.lock before hashing;
all394 hashes remain unchanged. Frozen source2d72b26d is evidence-only beyond
07718f07. Server06f9d1d0...58935 differs from the first multi-bin server ELF, so its
own fresh public/HTTP gates remain required; both artifacts are preserved. Root
froze it as c080-physical32. No CPU test was repeated because source did not change:
131 package tests and server check already passed with0 filtered.

No serving quality, throughput or queued>32 latency result is inferred here.
