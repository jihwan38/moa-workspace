# Frozen q022 quality-only handoff

Candidate: `/round2/candidates/q022-fixed-e4-kv-surrogate`.
Source: `9866221391bf463b6179b950ad26c63436de145c`.
Server:39,225,040 bytes, SHA256 `c0985c25bca1a0fa80ceb6ab81bda96bee09dd9787eac4fcfb880209f210c2d7`.
Captured immediately from successful a008 fresh-package/unique-output build, before candidate freeze. No mutable shared-target binary is used.

Read-only registry comparison finds302 identical common entries with verified ON decode/R32/partial model and300 with ON packed model, with zero differing common images. Probe-only whole-ring copy is diagnostic; server-only entries versus packed are the two greedy kernels already present in the decode model. All12 production codec-calling entrypoints are in the common set. Raw source differs from ON375bdcf4 only by two comments; original Cargo remains byte-identical. Peer scope review passed. Original OFF control used for numerical comparisons is fresha007; cached-ON a006 remains quarantined.

All14 paired structural routes and both precisions' head/scalar producer controls passed. Aggregate deterministic replay is95,682,560 logits and33,453,768,704 compared cache/sentinel bytes per precision (one comparison side). Cross-precision drift is material:175/365 logged top1 changes, mean0.2072 relativeL2 over128 samples per row. These repeated diagnostic trajectories are not task accuracy. R32 has no separate raw-cache snapshot; other disjoint routes provide byte gates.

Prepared public plan `state/q022-quality-plan.json` reserves r1360–r1365 for IFEval/GSM8K/MMLU/VQAv2/FLEURS EN/KO at10/17/17/6/6/6, matching q020. Root alone dispatches. Quality remains pending until those results, and no compressed-storage host port or throughput/adoption claim is made. The server retains BF16 pools with the fixedE4 roundtrip solely to test representation error across complete histories.
