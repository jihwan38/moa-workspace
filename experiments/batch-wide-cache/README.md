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
