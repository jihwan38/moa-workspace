P128 activation-cache isolation (R2-BWC01)
========================================

The bit-exact P128 FFN control was slower than eight B16 calls. Its outer
weight-row-tile loop repeatedly copied activations into all consuming slices.
This experiment changes two specific transfers separately:

* `batch_ffn_up_cached` loads P64×H activations once into H480 shards. For each
  B8 contraction it gathers the eight H shards with Broadcast1 into eight valid
  copies, selects the first copy, and uses CustomBroadcast256 directly into TRF.
  The first source is valid because every gathered copy contains the same H
  vector. Weight tiles, dot products, and downstream FFN code are unchanged.
* `batch_ffn_down_cached` stores all64 local L1920 input rows once (245760 bytes
  per slice). It releases the full decoded E4 shard by loading24 FP4 rows per
  weight tile and decoding four rows at a time into the existing BF16 tile.
  This preserves aggregate FP4 HBM bytes while avoiding five input repartitions.
* `batch_ffn_cached` combines both changes; canonical `batch_ffn` is unchanged.

A direct H480-to-all-slices switch failed: the eight-source switch only exposes
8#256 valid output slices, while forcing a256-source switch triggered a compiler
sequencer assertion. Those source versions and compiler logs are retained.
The two-stage gather/broadcast and cached-down variants compile successfully.

`test_wide_cache` checks all491520 outputs of each variant bitwise against the
validated control, then samples control/up/down/both/both/down/up/control.
Every sample performs three launches and one identical P128 read, divided by
three. Compile/build passed; hardware equality and timing are pending root.

A second up/gate candidate, `batch_ffn_up32`, avoids Switch broadcast. It caches
32 local rows (245760 bytes), loads/dequantizes each12-row FP4 weight tile, and
serves four B8 contractions from that cache. Two such superchunks cover the64
local rows. This is the largest power-of-two cache fitting alongside the BF16
weight tile and scratch;64 local rows alone need491520 bytes. Compared with the
initial P128 kernel, aggregate activation DM writes fall by five, while original
weight bytes/dequantization are performed twice per128 rows. The combined
`batch_ffn_up32_down_cached` keeps the same cached-down variant.
`test_wide_cache32` uses the same exact-output and symmetric timing protocol.
Both variants and their executable compile; hardware results are pending root.

Hardware results (root Arena317662 and317676): every candidate checked491520
outputs bit-exact against the original P128 FFN. The two-stage Switch candidate
was rejected: up-only157.4ms and combined153.46ms versus12.88ms control. Cached
down alone reduced runtime to9.13ms. The non-Switch32-row candidate then measured
approximately8.52ms up-only,9.02ms down-only, and4.59–4.63ms combined, against
approximately13.0ms control. The combined implementation is now canonical
`ops_batch_wide::batch_ffn`; the initial repeated-staging implementation remains
`batch_ffn_staged_control` for reproducible comparisons. No approximation or
weight change was introduced. Whole-model runtime validation follows in root.
