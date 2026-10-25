# Fixed E4 KV complete B16 micro: PASS

Root runs r1289/job321461 and r1290/job321462 completed in16.9067s and16.9588s.
Functional source27e9bde8, frozen ELF
`0bcaf01a375418d4dde5d1d4aeffddc2d88dc0bec6d0650ad234b6ad01b2e9a0`.
The parent candidate/reader/commit and BF16/O control modules remain unchanged.

Complete one-launch append + attention + supplied O takes **238.778→201.287µs**
on average (**15.701% less time**,37.492µs saved). Four cases, each including two
feedback steps and ABBA12 completed calls per round:

| Case / supplied O / live rows | BF16 µs | Fixed E4 µs | Reduction |
|---|---:|---:|---:|
|0 / layer0 /16|238.174|201.480|15.406%|
|1 / layer6 /14|239.144|201.233|15.853%|
|2 / layer46 /16|238.417|201.737|15.385%|
|3 / layer0 /14|239.378|200.698|16.159%|

Both jobs pass the predeclared aggregate10% complete-call gate. Setup, uploads,
allocations, metadata, downloads and CPU checks are outside both arms equally.
Append conversion/scatter and all E4 decode/materialization costs are inside.
The control is the current composed BF16 graph, not the old three-launch graph.

**Exact comparisons:**2,031,616 direct intended attention/O bytes plus2,031,616
post-timing output-invariance bytes;17,716,740,096 whole-cache comparison bytes;
78,679,872 immutable Q/K/V/offset/mask/O-weight bytes. Counts are repeated checked
bytes, not unique cache capacity. CPU scatter references cover all64 slots,
untouched and future finite poison, sparse slots0/31/32/63, positions including
256/1023/1024/8191, physical wrap on the next append, inactive NaN K/V rows,
matched feedback and a bijective live-owner reversal. Inactive outputs remain
zero. All intended results are bit-exact to BF16 attention/O applied to CPU E4
roundtripped cache and append values.

Each job also checks all65,280 distinct finite BF16 inputs through the real
append graph. The two32,768-value tensors replace the256 nonfinite encoding
positions with signed maximum-finite BF16, so30,782 supplied entries exceed448.
Every encoded byte matches the independent nearest-even lattice oracle:
finite overflow saturates to signed448, signed zero is preserved, and midpoint,
subnormal/underflow and range-boundary cases pass. This establishes these **finite
input** device semantics; NaN/infinity encoding semantics are outside the gate.
The standalone CPU oracle additionally checks all E4 midpoint neighbors and all
finite BF16 patterns independently of the SDK conversion function.

**Original BF16 drift is separate:** synthetic attention relative L2 ranges
0.023864–0.136314; supplied-O output relative L2 ranges0.024371–0.125196.
Maximum absolute differences are1.664063 attention and2.492188 O. No active
nonfinite results occur. Q/K/V and history are synthetic; only O weights are
provided-model data. These are not real-model history/quality results and do not
justify serving activation.

Actual cluster0 profile means (raw cycles, one complete graph per case):

| Span | BF16 | E4 |
|---|---:|---:|
|Task|458334|387412|
|DMA union|353638|251635|
|TU union|169096|231236|
|Table load union|5380|26895|

DMA count increases30→34, TU40→46, table loads1→5. Cluster1 has the same trend:
Task457949→386176, DMA union344757→243614, TU169120→231355. The net wall win
therefore includes the extra fixed-code decode work; halved cache payload does
not imply halved total latency. Span sums/unions overlap across engines and are
not additive savings or assumed clock-frequency conversions. The observed
complete-call result is the performance gate.

`collect_actual.py` parses the preserved logs and all profile groups. `actual.json`
contains both jobs' timings, exact counters, drift and both-cluster aggregate
profiles. No host/cache-storage port was made.
