# HOLD: native old-history QK does not shorten the complete P128 chain

The same-source compiler screen is123,624 cycles for current c079 BF16 versus
123,627 for the native-E4 experiment: effectively unchanged. Neither graph has
DramStack. The predeclared10% complete-chain gate fails. No NPU test, scaled
quantizer, cache shadow or host activation follows this screen.

Source `7ec9a1ae` explicitly casts the four local headed query rows into E4 DM
before the original ring broadcast, and casts gathered BF16 keys into E4 DM
before the TRF load. This is unscaled arithmetic and may saturate or underflow;
it is deliberately not a numerical candidate. First-chunk/current-chunk QK,
BF16 PV, mask/softmax/denominator, prepared heads and commit ordering are unchanged.
The initial unsupported BF16-to-E4 Fetch Adapter compile is retained as r1206.

The old-history QK TU shrinks4,364→2,316 estimated cycles. New K/Q casts cost
1,288/392 TU cycles, and the complete graph's later work dominates its end time.
The original QK TU alone is only3.53% of this complete resource span, even before
considering overlap. This does not establish a universal attention lower bound:
changed K storage or a different complete attention algorithm is separate work.

`analyze.py` verifies both run source hashes, no hidden stack, schedules and
operation spans. These compiler spans are not wall time or actual-device speed;
summed resource durations overlap. Prior R64 native-QK evidence is not reused as
P128 numerical or performance proof.
