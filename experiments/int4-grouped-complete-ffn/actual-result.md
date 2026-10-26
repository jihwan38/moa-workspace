Actual job **323213** passed intended-code correctness and failed the preregistered
**>=10% complete-chain gain in both row orders**. HOLD; no model integration,
quality claim or serving promotion follows. Runtime was13.598614656s with exit0;
exit0 denotes correctness, while the final performance marker is false.

| Row order | Current production µs | Seven-call candidate µs | Gain | 10% gate |
|---|---:|---:|---:|---|
| 0 | 1067.064813 | 1089.835000 | -2.133909% | FAIL |
| 1 | 1064.236250 | 1086.606000 | -2.101953% | FAIL |

Each mean has16 completed chains/arm:8 restored-input samples/block, candidate–
production–production–candidate. Every raw validation, seven awaits, canonical
BF16 write/reread, packed/offset/scale store and tail is included. Restores,
startup, downloads and oracles are outside. The control is the unchanged q020
one-call production graph, with the exact same embedded image.

Correctness reconciles196,608,000 intended BF16 comparison events,6144 independent
i64/F64 samples,78,684,160 permutation values,40,960 scale bits,40,960 relative
offsets and19,660,800 packed bytes. Full reuse checks covered226,271,232 bytes;
bytewise immutable checks covered599,335,048 bytes. A separate394,728,988-byte
full-read FNV fingerprint check protects production weights/norms. These counts
are not distinct model logits or task answers. All512 rows and four/16 groups
per row are covered. Both row orders reproduce the same printed drift/zero metrics.

The fixed-entry, step0 output L2 versus current E4 is0.3963337423 joined,
0.4820915604 DOWN and0.4238808191 final residual. The old ungrouped complete
experiment reported0.5744822058 joined and0.8466086464 final residual on the same
provided layer0/synthetic entry fixture; all three weight encodings/fingerprints
match exactly. This comparison changes both activation grouping and scale
placement, so it is not an isolated additive error decomposition or model-quality
measurement. Step1 uses independently evolved residual feedback and remains
separate in the machine-readable report.

On the candidate's own BF16 prefix, A4 has69.4473266602% zero codes and scalar
RN32 reconstruction L2=0.427417544056. Its own GeGLU BF16 hidden has84.4386927287%
zero codes and L2=0.2895428266901. Current E4 zero fractions are2.2115580241% and
18.2193247477%, respectively. Only the step0 prefix BF16 input is matched across
these quantizers: the hidden input already differs through approximate UP/GATE.
Step1 inputs additionally differ through residual feedback. Scalar reconstruction
is not the projection dot-product error.

The eight separately restored phase-clock repeats report:

| Completed-await phase | Mean µs | Observed range µs |
|---|---:|---:|
| prefix_us | 82.197500 | 79.470000–92.852000 |
| input_quant_pack_us | 51.727375 | 50.551000–52.481000 |
| upgate_us | 359.291125 | 358.077000–360.086000 |
| geglu_bridge_us | 187.026875 | 185.664000–188.333000 |
| hidden_quant_pack_us | 95.238375 | 94.112000–97.491000 |
| down_us | 261.163125 | 254.784000–267.644000 |
| tail_us | 51.138125 | 50.321000–52.950000 |

Mean phase sum is1087.782500µs; the enclosing total is
1088.127875µs, including0.345375µs mean clock gaps.
The separately timed one-call control here averages1102.783125µs,
which differs from the ABBA control. These phase records cannot replace the
failed ABBA gate. No profile environment was supplied; arena-records is empty.
The measurements do not establish per-op device costs, a constant launch cost,
a warmup cause or costs obtained by subtracting standalone component times.

Startup records are6,009.686681ms for provided-BF16 CPU encoding **and audit**,
56.527713ms for original production prepack/join, and178.150341ms for candidate
**plus intended-reference** uploads. This is not an optimized candidate-only or
48-layer startup estimate. Declared retained payload is774,022,688 bytes,
including63,979,520 candidate scratch bytes with8192/32768-byte offset owners;
it excludes temporary oracle buffers, CPU copies, allocator and kernel DM/TRF.
It is not server peak or model-fit evidence.

Build source dc3524a3, submission source0a7863f3, ELF
`3444ef561e2d39d6a15ec6340548d07e65b873d2884b62ac04f9ea2ccd66cdd4` (6,550,888 bytes).
All421 submitted/build/current source/config hashes match. All19 images match
prior actual gates:13 original complete,2 grouped quantizer,4 grouped projection/
reference; the q020 whole-graph control also matches its frozen server image.
Raw log SHA256 `e75b591702df11d0fa72929fa920864509101d591e0104066ef089584b371ac6`.

Replay with `python3 experiments/int4-grouped-complete-ffn/analyze_actual.py
--verify-local-artifacts`. Raw run/status/log/source hashes are in
`actual-323213/`; every timing and phase record, including ranges and failed
gates, is retained in `actual-result.json`. No source or build changed after
capture. Full model, task quality, padded tails, full-attention prefix, HTTP and
official score remain untested by this probe.
