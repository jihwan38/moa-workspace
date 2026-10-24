# Generic greedy reduction contract

`ops_greedy_r64::greedy_argmax_r64` consumes exactly the blocked BF16 logits from
`ops_r64_io_blocked::logits_batch`. No model input, benchmark, token identity, or
request content changes the computation. It evaluates all 64 physical output
rows; the host consumes only the live prefix.

Inputs after `device`:

- `logits: &HbmTensor<bf16, Chip, BlockedLogits>`.
- `ban_words: &HbmTensor<i32, Chip, BanWords>` where
  `BanWords = m![W / 32, R64]`. Host flat word index is
  `(token / 32) * 64 + row`; bit `token % 32` means banned. Ignore out-of-range
  token IDs while packing. Full physical upload is 2 MiB.
- Reusable `negative_token_ids: i32[W] = [0, -1, ..., -262143]`.
- Reusable `bit_positions: i32[W % 32] = [0, ..., 31]`.
- Mutable `out: i32[GreedyResult]`, with
  `GreedyResult = m![R64, GreedyField]` and `GreedyField = 8`.

Output flat offset `row * 8`:

0. Maximum allowed signed ordering key, or `-65536` when every valid token is
   banned. Every finite value's key is strictly greater than that sentinel.
1. Negative token ID of the lowest-ID allowed maximum.
2. Negative token ID of the lowest-ID unmasked maximum.
3. Maximum absolute BF16 code. `>= 0x7f80` means a nonfinite input exists: use the
   original CPU sampler for this row. This includes infinities and all NaNs,
   regardless of whether that token was banned.
4. Unmasked maximum signed key, used to apply the zero/subnormal fallback after
   the all-banned fallback.
5–7. Unspecified; do not consume.

For a finite row, let the chosen key be field4 if field0 is `-65536`, otherwise
field0. If this key is either `-1` or `0`, use the original CPU sampler: hardware
BF16-to-F32 fetch flushes subnormals, so a zero maximum is ambiguous. This also
conservatively falls back for an exact zero maximum. A normal selected maximum
cannot be displaced by flushed subnormals, which always map to ±0 below every
positive normal and above every negative normal.

Otherwise select `-field2` if field0 is `-65536`, otherwise `-field1`.
This exactly preserves the deliberate all-banned unmasked fallback. Validate the
selected token in the shared Generation transition API before advancing state.
Only invoke this path for `SamplingConfig::is_greedy()` (`temperature <= 0.0`).
Top-k/top-p are immaterial on that branch; every stochastic configuration uses
its existing sampler.

Normal finite signed keys preserve Rust `f32::total_cmp`, including sign ordering, and the
second integer maximum implements lowest-ID ties. Any unsupported nonfinite or zero/subnormal-maximum
numeric ordering is explicitly deferred to the original host conversion and
sampler; it is never silently replaced with a floating-point Max reduction.

Actual NPU job318061 passed this ABI on all64 testrows, including45 exact device
selections and19 deliberately triggered fallback rows. See `actual-318061.json`
for source/binary provenance and timing. Full-model Worker integration is a
separate validation gate.
