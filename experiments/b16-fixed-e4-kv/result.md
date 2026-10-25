# Fixed unscaled E4 sliding KV resource screen

The complete current live16 append → sliding attention → original O graph
decreased from244,589 to204,984 estimated cycles (16.19%). Both graphs retain
the same32-byte index bounce allocation; no additional DramStack appeared.
`analyze.py` verifies all394 source/config hashes and both captured schedules.
This passes the declared10% compiler gate, not a hardware or quality gate.

The source recipe starts at c079/root `fb44a66d`, keeps64 physical pool slots,
and changes their storage dtype to unscaled E4 in this isolated graph only.
Appending BF16 K/V converts through F32 to E4. Reading widens E4 with exact
BF16 lookup, then retains the original BF16 QK, softmax, PV and O operations.
Compared with archived adaptive E4 storage, there is no per-token/head maximum,
scale array, division, scale transfer or reconstruction multiplication. E4
overflow and underflow are unproven approximation risks. Production is untouched.

Final source `4bf039b2` materializes both decoded K and V in real BF16 DM before
their original consumers. The first Sub-context LUT prototype `e0fbe296` failed
because SDK0.9 lookup is Main-only. The padded V LUT/transpose prototype failed
Collect-time validation. Both failures remain in r1262/r1264/r1271; the final
graph does not claim either fusion. K/V decode passes and all table loads are
included. Gather duration sums decreased115,804→59,388, while TU sums increased
85,195→115,447 and table-load sums1,293→6,465. These durations overlap.

E4 byte offsets are half the BF16 offsets: a token row is2,048 bytes and a whole
ring is2,097,152 bytes. The future micro must independently verify code bytes,
intended BF16-roundtrip attention/O, full pool mutations, live-prefix poisoning,
sparse ring ownership, wraparound and actual complete-chain timing. Original
BF16 drift must remain separate. No host/storage integration is authorized by
this resource result alone.
