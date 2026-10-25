# Exact two-live-quarter tail diagnostic

Base9d6a932. No Worker dispatch changes. Actual r796 tails were one owner256 or two owners128+128, dispatched as two complete P128 passes. Archived P256FFN/r194 and pair model/r250/r268 passed against older kernels but do not establish a win over current production.

Dedicated PreparedTail256 owns exactly two real PreparedWide states and one/two actual owner mappings. Single256 derives chronological halves; pair128+128 validates independent rings/pages. Existing four-owner and Prompt512 constructors remain unchanged. A shared private layer executor takes only actual states for attention/commits/cursors, while the current four-quarter norm/QKV/O/FFN tensor kernels receive two explicitly initialized padding quarters. Padding has no Sequence, Blocks, positions, logits or cache writes. All reduction axes are H/L; token/quarter axes are retained. Normal padding is zero; a scoped diagnostic fills padding with a chosen BF16 sentinel to test live-output independence.

One complete layer remains the scheduling checkpoint. Active allocation guards expose only actual owners; other prompt scratch is unavailable until completion. Cancelled or failed in-flight calls retain existing discard-Model/Workspace semantics.

Gate: oneModel/sharedweights, old two native P128 passes versus new tail, all-vocabulary exact logits and subsequent cache reads; starts0/256/896/1920, chronological ring/page wrap, two independent owner orders,48 R64 interleaves, reused group scratch and poisoned inactive tensors. Inclusive preparation+layers+finish timings exclude D2H and interleaved unrelated calls equally. No speed assumption from padded work, no automatic HTTP activation.


Pre-device review: e2e_evidence found the initial diagnostic's shifted padded HBM views repeated an Arena317917 BufferTooSmall failure. Before any device build, the reader was replaced by the exact `host/kv_diagnostic.rs` raw-byte whole-pool+byte-offset helper from bd6721b. It compares every sliding position and every reserved full page without modifying pools; this avoids slow typed host conversion. Ownership/control review otherwise found no concrete defect. All91 host tests passed after the replacement.

The probe accepts optional case0..8 (or `all`, default) so the coordinator can split expensive full-cache/R64 gates within the Arena deadline. Starts include the observed same-owner1792+256 tail and1920 ring/page wrap. Every selected case also runs the reversed-owner reset. Timing includes padding initialization, both actual embedding preparations, all48 layer calls and final output computation; D2H, full-cache scans, and unrelated R64 interleaves are excluded from both model timing totals. No speed claim exists before actual hardware.
