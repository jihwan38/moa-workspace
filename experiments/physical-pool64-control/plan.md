# Physical64 reference for the fixed32-owner probes

Base c079394d60a2 retains physical Rc64=64 and Worker capacity64. The same
07718f07 diagnostic-only patch is imported as4b2e025c; the final Group PASS label
is corrected to physical64, without changing inputs, ownership, arithmetic,
selectors, checkpoint order, hash bytes or comparisons. History creates a
Workspace with32 logical request slots in both programs while old control still
allocates64 physical slots. Therefore its invalid slot32 guard remains meaningful.

The micro compares each program to its independent affine CPU oracle, including
all allocated zero/untouched bytes. The model programs emit SHA256 of exactly the
same logical logits and reserved-cache bytes. A separate root-dispatched run for
each program is required before reporting cross-program digest equality. These
are digest comparisons, in addition to each program's in-process exact pair tests;
they are not an in-process physical64-versus32 full-byte comparison.

No NPU dispatch, root edits, model arithmetic changes or serving activation.
