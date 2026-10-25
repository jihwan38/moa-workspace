Source basece88461, persistent micro3124649, complete consumer chain0df4516.
Root actual318986 proves every intermediate byte and repeated state for both
attention types; its actual trace and results are recorded ine132085.

GroupPrefill now owns one3.75MiB blocked residual. PreparedGroup owns the storage
selector. It is effective only with default norm/output/quantizer choices; the
fused FFN core selector remains independent. Existing original norm controls keep
canonical storage for the complete execution, so their future timing comparisons
include storage fallback. original_blocked_residual is the primary isolated
control: all other kernel choices match.

At layer0, after all four WidePrefill preparations, pack every row exactly once.
All QKV norm and postattention/post-FFN residual consumers then read/write the
blocked state. Canonical row.x is deliberately stale during intermediate layers;
attention helpers return at defer_output before accessing it. At the final layer,
unpack all four quarters exactly once before any cursor advances to completion.
The unchanged owner finish paths therefore read current canonical values: four
owners, Prompt512 quarter3, and Prompt384128 quarters2/3. Future pair metadata can
select any owner-final quarter because all four are restored.

No ordinary Wide/Prefill/R64 ABI or scratch ownership changes. Both independent
and chronological cache commit sequences remain unchanged. An active group keeps
both cache reservations and the grouped residual buffer; independent decoding has
separate scratch. On error/cancellation the existing discard-model/workspace
contract applies. Next preparation initializes all four canonical rows and the
first advance overwrites every blocked-state value before any read.

Checks: server+three diagnostic probe cargo checks and69 host tests passed.
Batch_kernels independent read-only review found no stale canonical read,
owner-output, scratch or selector defect. Actual whole-model/public gates remain.
The three probes compare the same grouped arithmetic with only storage changed;
initial/final conversion costs are inside timed model execution, not omitted.
