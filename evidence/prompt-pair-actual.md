Parent NPU r250 (oldtransfer source15a4cf9) passed both full262144-logit vectors
bit-exact for two128-token prompts at start0. Uninterrupted two-P128 control
297.653ms; paired48layer quanta279.373ms, maximum10.829ms. First lazy pair
allocation+embedding14.760ms was separate, so these numbers are not a warm
full-request throughput result. FFN77.017ms; pack6.607ms; unpack6.194ms.

Row-contiguous transfer source5518be4 passed r257:983040 BF16 raw values exact
for both original/swapped halves. MatchedABBA old pack120.694–131.435us versus
candidate18.513–19.478us; old unpack122.469–130.169us versus18.528–18.692us.
Only PairPrefill's IO module changes here; QKV/O/attention/FFN arithmetic stays
on the frozen baseline for an interpretable comparison. Paired automatic Worker
dispatch remains off. Boundary/interleaved fullmodel proof and post-prompt KV
reuse are still required, now with the validated fast canonical zero constructor.
