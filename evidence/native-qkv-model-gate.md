The runtime-only export e2f16dc adds two modules and lib registrations only.
This separate source-only diagnostic hook preserves BF16 as the default.

WidePrefill owns one E4[P128,H] input scratch and one F32[P128,1#8] scale
scratch (495,616 bytes). Its normal/resumable calls retain BF16 QKV. Explicit
Model::forward_sequences_native_qkv uses WidePrefill::run_native_qkv for a
complete prompt chunk: the precision selector belongs to its PreparedWide state,
so Result errors, panic unwinding or future cancellation cannot leak mode into
later requests. Exclusive &mut Model/WidePrefill ownership precludes concurrent
calls; existing active-prefill allocation guards remain.
Only the input-normalized Q/K/V projections change. Heads, RoPE, attention,
KV commits, output projection, FFN, residuals and logits are unchanged.

compare_native_qkv TOKENS SALT (defaults513,0) loads one model and one canonical
pool, with independent control/candidate rings62/63 and disjoint high physical
pages. Every full128 chunk uses current BF16 QKV control versus native candidate;
B16/scalar tails use identical existing math. It reports all-vocabulary finite
logits, exact count, max absolute error, relativeL2, argmax, top10 overlap and
model-call timing, followed by four identical teacher-forced decode tokens to
expose persistent KV numerical changes. It does not assert artificial equality
or a quality threshold for this deliberately approximate computation.

cargo check --bin compare_native_qkv passed. This source has not been built
into a device binary or run on NPU. Public accuracy and mixed media suites are
still necessary; a synthetic logit comparison cannot establish task quality.

Read-only attention audit found the initial persistent selector was not safe
against future cancellation. The follow-up moves it to per-execution PreparedWide
state; all normal prepare calls initialize BF16. No arithmetic or dispatch policy
changes, and no kernel/binary result is claimed for this host-only repair.

Explicit forward_sequences_bf16_qkv forces only the P128 QKV precision toBF16,
independent of a future candidate's ordinary default. The paired probe uses
this method for128-row controls and explicit native for candidates. All source
defaults remain BF16; root owns any later isolated public-candidate flip.
