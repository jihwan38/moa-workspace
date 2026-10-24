# One chronological512-token prompt

Source base: `9616852` (validated independent four-prompt host and native P128
QKV/FFN arithmetic). This experiment adds an explicit Model diagnostic only;
Generation chunk selection and Worker dispatch remain unchanged.

`start_prefill512` validates one canonical allocation,512 input rows,128-aligned
start, sufficient distinct physical pages, context bounds, and final-only logits.
A distinct `PreparedPrompt512` constructor derives starts s/s+128/s+256/s+384 and
keeps one public owner. It does not weaken the four-request independence guard.
Each layer projects all four previous-layer hidden buffers, then processes their
attention and KV commits chronologically. Sliding attention reads the old ring
before each128-row commit; full attention commits only the current chunk and
visits pages through that chunk. Shared blocked FFN then updates all four hidden
buffers. Only chunk3 produces the final requested logits. The scheduler may yield
only after a complete layer; errors/cancelled in-flight futures require discarding
the Model/Workspace exactly as for other resumable prompt execution.

Expected removed work: repeated QKV weight loads and repeated FFN weight/input
traffic across four sequential128-token passes. Existing kernels and arithmetic
are unchanged. Exact full-model parity is an expectation, not a result yet.

Gate: `compare_prompt512 START INTERLEAVE`, with START0/384/896 to cover an empty
history, a full-page crossing, and a sliding-ring crossing. Use one canonical pool
with disjoint control/candidate ring/page pairs and reversed physical page IDs.
Compare every final logit against four sequential native P128 calls, every logit
of four teacher-forced post-prompt steps, and optional48 paired disjoint R64
interleaves. Verify the active owner list contains one ring and rejects ring/page
aliases. Host tests preserve independent-request rejection and validate supported
alignment/count/final-logit constraints.

Timing reports include embedding/metadata preparation and every complete layer
for both paths. Group scratch allocation occurs once before either timed path,
matching Worker startup allocation. Report maximum layer quantum separately.
No performance conclusion, HTTP activation, or generation512 dispatch is justified
until actual NPU parity and complete-time measurements pass.
