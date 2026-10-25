# Actual319595: expand the original three-stage composition

The14/16 micro passed in11.211s. All **6,442,450,944 CPU-addressed full-cache
bytes** and **1,523,712 attention/output bytes** were exact across supplied O
layers0/6/46, physical permutations, wrap/reset and NaN inactive K/V inputs.
Every case began with separately poisoned candidate/control output buffers;
active output was finite, inactive output masked zero, and all Q/K/V input
bytes remained unchanged. The cache scans include all untouched slots and tokens.

The complete three-launch control took292.43–297.09us; one retained-stage graph
took235.48–240.18us across all six cases. At live14/layer0, ABBA was
293.572/292.502us versus237.479/235.483us. Each number averages12 completed
chains, excluding identical metadata/readbacks. The second consecutive candidate
batch is slightly faster; both candidate batches beat both controls.

Actual live14 cluster0 Task sum falls466,773cycles over three Tasks to458,003
cycles in one Task. DMA count remains30 and TU count40, with essentially the
same traffic and arithmetic durations. Live16 similarly falls467,185→456,517.
Thus most of the measured wall reduction comes from eliminating completion/
dispatch boundaries; it is not a reduction in cache bytes, precision or O work.
Overlapping category unions must not be summed as elapsed time.

Proceed to compile/test all live1–16 variants, then a default-off per-call
whole-model selector with matched all-vocabulary/full-cache comparisons. Do not
activate serving based on this micro. Forty sliding layers times the measured
roughly55us saving suggests about2.2ms per B16 step before model overhead, an
estimate only. The distributed-commit experiment remains rejected; this graph
uses the original one-live-slice commit exactly.

Frozen tested source4629e18, ELF SHA
`5a7437ae8fa2234efc758f1b8cd78ef1dfa11d3f38d8abbcb213012df43eea68`.
Build a003 provenance and actual r925/319595 logs remain unchanged.
