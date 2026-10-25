# Dedicated two-owner384+128 prompt checkpoint

Base is the passed P512 core integration `ae0ddbd` (root base4f0be45). Scheduler
trace r672 seq84–87 had owner38 with384 prompt inputs and owner44 with128; four
P128 calls consumed about378ms device-call wall time plus decode interleaves.
This is an input-shape opportunity, not benchmark-specific behavior.

The new `PreparedPrompt384128` wraps private four-quarter execution state.
`GroupPrefill::supports_prompt384128` validates exactly two independent original
owners, lengths384/128,128-aligned checked position ranges and final-only logits.
Only then does the constructor derive A0/A1/A2/B0 internally. The original
four-independent-quarter guard is untouched and still rejects these duplicate
A metadata entries. Existing per-layer attention/commit order provides each
owner's causal history; all dense kernels and numeric boundaries are unchanged.

`Model::start_prefill384_128` owns a distinct active field. Active-allocation
reporting exposes exactly A and B, so alias/scratch protection covers both.
`advance_prefill384_128` advances one complete layer and finishes quarter2 for A
and quarter3 for B; it returns exactly two owner-ordered outputs. Only final
inputs requesting logits yield valid logit contents, as in existing APIs.
Completion clears the active field. Failed/cancelled in-flight device futures
have the existing discard-Model/Workspace contract, not a new retry behavior.
No Worker routing, admission, reservations or Generation progress changes yet.

All69 host tests passed. New tests cover chronological derivation across page/
ring boundaries, unchanged ordinary independence rejection, wrong sizes,
alignment/overflow, intermediate logits, duplicate rings/pages and internal
page duplication.

`compare_prompt384_128 [A_start=1664] [B_start=1920] [interleave=1]` rebuilds both
actual causal histories in disjoint rings and reversed physical pages. The
reference executes four original native-P128 passes; the candidate runs one
four-quarter checkpoint. It checks all vocabulary bits for both owners, both
owner alias guards, shared-scratch rejection,48 independent R64 interleaves,
four two-owner B16 decodes (default crosses2048 into the fifth full page), and
position-zero reuse with new prompt contents. Timing includes all completed
model calls but excludes host downloads and unrelated decoder calls. Hardware
execution remains coordinated by root; no speed or model parity claim precedes
that gate. Worker integration follows only after the constructor/model proof.
