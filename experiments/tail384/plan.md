# Tail384 model-only gate

Base650642a (root c066 source with exact Tail256 Worker, head fusion and media).
Dedicated PreparedTail384 owns exactly3 real PreparedWide states and explicit
owner first/final row maps; only one tensor quarter is initialized as padding.
No Worker, device kernel, source weight, original Group constructor or Tail256
change. The common private layer executor now accepts a state slice of length3.

Accept one384,256+128,128+256,or128+128+128 with aligned checked ranges,
final-only logits and independent real reservations. Private repeated-owner
quarters preserve chronology. Model active allocations expose only real owners,
shared scratch stays excluded from other prompt work, and independent decode
may execute only between completed layers. Failed/cancelled in-flight execution
retains the existing discard-Model/Workspace contract.

Probe compares all provided-model vocabulary logits and every raw cache byte
with three current P128 calls. Separate true histories, NaN padding, reverse
owner order/reset, sentinel cache, ring/page boundaries, teacherforced decodes,
and48 R64 interleaves are required. Timing includes preparation/uploads/padding
allocation,48 complete layers and finish; downloads/interleaves/cache scans are
excluded consistently. Cases are independently selectable to stay below job
limits. Parent alone dispatches NPU; compile success is not a speed/parity claim.

The oracle also measures the CURRENT Worker decomposition, Tail256 followed by
one P128, for every legal owner shape. It independently compares that route's
all-vocabulary outputs to the three-P128 control, recomputing the identical true
prefix before the second control. Both controls include their own preparation
and finish work and exclude reads. Candidate claims must use this two-plus-one
inclusive baseline, not only the older three-call baseline.

Host checks:101 library tests passed. Probe check passed after correcting the
adapted interleave reference slot from4 to6 before any device build/run. Source
scope is only model.rs, prefill_group.rs and the new compare_tail384 probe;
Worker source and device modules remain identical to650642a.
