# Bounded source review and diagnostic scope

Root and round2_path independently reviewed `f7599e4b`: no ownership or dispatch
blocker found. Shadow construction retains a local Vec until all 48 layers finish,
then publishes one Option. Failure/cancellation before publication drops the local
owners and leaves the originals and `None` intact. Packed HostTensor owns its
buffer through upload. This is a source lifetime argument, not a cancellation
stress result. Check a001 passed in 14.65 seconds.

Both F4 diagnostic selectors normalize to Automatic before width, QKV, O, R32
and retained-output policy. Only independent canonical B16 rows with staged and
composed FFN borrow the immutable F4 layer references for one awaited invocation.
Scalar, prompt/mixed, R32, original B16 LM and projection-owned prompt FFN remain
on their original paths. Current enum only includes PowerOfTwoRange; MSE remains
outside this scaffold pending a new actual kernel gate.

`compare_b16_f4 model 512` runs three schedules: original in physical half A,
candidate in B, then candidate replay in A. Sixteen logical owners have distinct
real 512-token histories; one P128 checkpoint advances through 48 interleaved
quanta. Widths 1/2/8/16, row permutations, reset-to-zero, an explicit old-epilogue
fallback and F4 re-entry are included. Every touched cache coordinate is rewritten
on replay; untouched storage remains the equal zero initialization. Final replay
compares all 16 full sliding rings and reserved full pages against candidate B.
Original-to-F4 logit drift is reported with no exactness or quality assertion.
Candidate replay, unchanged scalar/Automatic/P128 outputs and own-output device
greedy are strict. Seeded stochastic equality is asserted only for exact replay.

`compare_b16_f4 guards 512` separately recomputes 17 real histories twice and
compares original versus explicit candidate on scalar and R32 calls, for which
F4 must not dispatch. This split bounds each hardware job and avoids requiring
34 physical slots in the 32-slot workspace. Both modes first allocate all current
group/projection storage, verify missing-shadow rejection, prepare all 144
projections, verify idempotence, and report actual device capacity and separate
read/encode/upload timings. No performance or quality claim exists before runtime.
