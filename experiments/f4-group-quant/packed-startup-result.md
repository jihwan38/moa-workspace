# Packed-only startup helper: CPU gate PASS

Functional source `058f785b` adds `quantize_blocked_e4_packed` without changing
the original `quantize_blocked_e4` body. It returns only packed bytes and group
scales. One pass finds each group's maximum and rejects NaNs; a second pass
writes two table-selected nibbles per output byte. The large packed output is
allocated only after validation. The dense E4 diagnostic reference is omitted.

The original helper already used a19×256 integer lookup table; it did not run a
per-weight F64 nearest-code search. The B16 diagnostic's earlier
`cpu_encode_and_audit_ms` also included its every-value F64 error audit and is
neither encode-only nor startup time.

Both helpers match the independent Fraction proof on1,843,200 blocked/permuted
values and1,920 group scales. The original4,826 finite E4/scale decisions remain
checked. Both reject the same invalid geometries and signed NaNs. A separate
read-only review of058f785b found no indexing/rounding change.

Standalone `rustc -C opt-level=3`, one i7-13700 CPU thread, fresh-process ABBA,
five calls per arm, one58,982,400-byte15360×3840 source:

| CPU helper |10 calls, elapsed range|Median|Peak RSS|
|---|---:|---:|---:|
| Original, including dense E4 reference|81.46–129.94ms|89.65ms|146,116KiB|
| Packed-only|12.73–17.52ms|13.02ms|88,360KiB|

Packed-only median source throughput is4.53GB/s. The last control arm was slower
than the first; all samples are retained rather than presenting a precise speed
ratio. RSS uses child `getrusage`, includes the59MB source,29.5MB packed output
and245,760B scales, and excludes other model/runtime processes. Timers include
encoding and output allocations; fixture generation, checksums, freeing outputs,
NPU transfers and the diagnostic F64 pass are outside. This is not an actual
model startup measurement or a task-quality result.

The standalone benchmark's first orchestration attempt completed all arms but
failed while asking git for a worktree SHA inside Docker (host `.git` pointer
is not mounted there). The corrected runner receives the host SHA explicitly;
the reported ABBA run completed successfully. This was not an encoder failure.

Minimal later startup experiment: retain original native E4/row scales, process
one projection at a time, download bytes, encode packed-only, upload owned F4 and
F32 group scales, then immediately drop both host buffers. Record each transfer
and encode separately; byte verification is untimed. Retained all48×3 shadow
payload is4,282,122,240B (before allocator padding), so the existing model's
allocator headroom must be measured before allocating them. P512 additionally
needs a joined-UP/GATE storage decision; borrowed padded-half aliasing is not
assumed safe. Rc32 capacity work is separate. No serving integration is added.
