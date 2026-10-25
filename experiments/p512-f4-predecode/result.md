# HOLD: P512 predecoded F4 weights

The complete graph compiled with no DramStack, but failed the declared combined
resource gate. Original E4 is1,076,934 cycles, prior inline F4 is917,695, and
one-time predecode is954,729:11.35% below E4, but4.04% above inline F4.
The earlier inline implementation already failed its actual10% gate with only
7.3–7.8% improvement. No NPU run or host integration is justified here.

`eb6af906` expands the packed F4 weight into real E4 DM once before the eight
UP/GATE or four DOWN query chunks. Its packed temporary ends before the chunks;
every later dot reuses the expanded weight. Per960 F32 scales still multiply
each local dot before the original inter-slice reduction. Quantization, GeGLU,
global factors, norm stores and the residual path are unchanged.

The change removes repeated LUTs but adds a full decoded weight allocation and
Main-engine decode pass. Compile success establishes legal storage, not reduced
latency. `analyze.py` verifies source hashes and captures operator sums from all
three same-tree schedules; overlapping sums are not a wall-time decomposition.
This is a closed dataflow prototype, not a rejection of every possible compressed
prefill design. The old inline hardware result remains preserved.
