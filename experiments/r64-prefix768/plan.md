# Exact unwrapped causal prefix768

Control: production `ops_request_sliding_r64` from root30038f5, SHA256
c9ab06f706963d5a78aaf1123b334ac8d55bd33596ce2885b5b81205268dd8d0.
Actual64-request512-history profile318724 spends29.815ms in40 sliding layers.
Positions512..519 cannot use the existing prefix512 route. Earlier actual
matching traces show16 K/V gathers around57k cycles each, about65.7% of1.395M
whole-task cycles. K/V are already shared across Gs2; R8 groups read disjoint
requests. Q and index spans are much smaller. There is no claimed2x KV duplication.

Use this general route only when all active positions are unwrapped and <768,
with zero visibility mask for every column>=768. Pool/offset/output ABI is
identical to the full kernel. Existing prefix512 remains preferable when eligible;
longer or wrapped histories retain the full kernel. No host dispatch is changed.

Read512 then256 keys as power-of-two payloads. Compute each original256-wide QK
dot and each32-key weighted-V partial unchanged. Materialize scores in the original
1024-element row, zeroing the omitted256 positions before the unchanged premax,
max, exp,1024-element denominator sum and EPS. Place the16+8 BF16 numerator planes
in their original32-plane order; explicitly zero the omitted8 and execute the
unchanged32-way F32 inter-slice reduction and BF16/divide boundaries.
Only small score/partial tensors are redistributed. K/V blocks do not incur an
additional DM redistribution. The third-quarter byte base is computed by exact
integer addition of512*8*256*2 to the existing physical ring base.

Compiler evidence and rejected variants are retained:
- Padded three-quarter gather indices fail the index-list verifier.
- Dense three-quarter indices plus24 live planes hit the DMA sequencer internal
  unwrap, isolated with gather-only code. This is a compiler failure, not NPU data.
- Two power-of-two gathers compile, but joining whole KV blocks adds large DMA
  copies (schedule3). Rejected before hardware measurement.
- Current score/partial-only join compiles (schedule5 estimated694160 cycles).
  Estimates are not actual device latency; the gate decides whether it wins.

Gate: original full kernel and original scalar-row equality for noncontiguous
slots at0,511,512,519,543,639,766,767; finite future poison through1023, sparse7-row
updates and64→7→64 transitions, full1024 masks, warm ABBA and actual profiling.
No benchmark answers, request sharing, dummy requests or weight changes occur.
A matched whole-model512-history gate remains required before any host adoption.
