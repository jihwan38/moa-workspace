Control873b5ea pairI/O actual317989/r232 passed all983040 rawBF16 values, but
pack121.548us plusunpack122.729us consumed244us of the~450us P256 FFN saving.
The control shards H into32 sections of120 elements, issuing128 strided240-byte
rowfragments per live slice. The candidate assigns one complete7680-byte hidden
row to each of128 live slices (padded to256), preserving all byte offsets and
rawbits. Expected removed work is fragmented TDMA; no model arithmetic changes.
Test every raw bit and swapped halves, then matchedABBA20-repeat pack/unpack
measurements. Keep host paired path on the old kernel until actual proof.
