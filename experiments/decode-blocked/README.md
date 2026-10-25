# Canonical decode blocked raw projection storage

Base `432d8b5` retains the root's current shared native QKV, blocked prompt
outputs, and prompt scheduling credits. This branch changes only canonical
R64/B16 decode raw QKV storage and its immediately following head consumer.

The four kernel modules are copied from batch-kernels `c80d699`. Arena318378
(r449) validated all three input cases, both widths, raw projection permutations,
canonical heads, and independent F64 projections. Complete quant+projection+head
R64 sliding was about .295→.208ms, full .274→.215ms; B16 sliding .200→.180ms,
full .206→.183ms. These are isolated kernel measurements, not model/SLO results.

`Prefill.q_raw_*` remains canonical: narrow prompt execution writes/reads it,
and the shared Prefill scratch can also serve Wide fallbacks. New
`decode_*_blocked_*` buffers are used only by `prefill_decode.rs` when the
workspace is canonical. Decode64 keeps an equivalent canonical reference set.
Additional raw scratch totals2703360 bytes (about2.58MiB) across B16/R64.
Canonical heads, attention, KV commit, FFN, LM head, media, and scheduling are
unchanged. No API replacement files were edited.

`Model::set_decode_blocked_output` is an explicit diagnostic storage-control
switch. Default serving enables blocked storage. Disabling it selects the
original same-precision projection/head functions; it does not reuse requests,
change arithmetic, or alter KV ownership. Noncanonical B16 retains the old path.

`compare_decode_blocked` fills512 real model-generated KV positions for each
of64 independent requests. It then compares canonical versus blocked storage
with physical ring/page remapping, live widths64/17/16/4, a full-page transition,
position0 reuse, and an interleaved15-token narrow prompt. Both passes use the
same dispatch widths and input histories so the gate isolates storage layout
instead of comparing different numerical implementations. It checks raw BF16
logits, reports per-stage completed execution time, and excludes host downloads
from model timing. Timing is diagnostic; this is not an HTTP/SLO or public
accuracy evaluation. Public C17 evaluation uses the corresponding server ELF.

Validation: `cargo check --lib` and the full release device build of both
`compare_decode_blocked` and `server` passed. ELF hashes are in `artifacts.json`.
The whole-model Arena gate remains pending; compilation does not establish
model equivalence, throughput, or public accuracy.
