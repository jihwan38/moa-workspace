# Original B16 commit → attention → H-split O composition

Base eb5004a; no distributed commit and no production dispatch.
Only VALID14 and16 device entrypoints initially. The original one-live-slice
commit, fullTs1024 attention, and K1024 BF16 partial-tree O bodies are retained
unchanged in stage helpers. All HBM input/intermediate/output boundaries remain.
Mutating current-token commit completes before attention reads the same Rc64
pools; attention output is retained before O. No numerical reassociation.

Gate supplied O weights and nonidentical random rows; candidate-first independent
NaN output scratch, live14 NaN K/V padding, physicalslot permutations and ringwrap,
full-pool independent CPU addressing/untouched neighbors, exact all attention/O
outputs and source immutability. Complete3→1 ABBA only; actual FURIOSA_OPT_PROFILE
trace. Original wrappers remain the control. No model/host activation until a
complete winner; no all1..16 graph expansion before that result.
Expected opportunity is two completion boundaries per layer, a bounded~1ms/40
layers hypothesis, not the rejected historical39us commit attribution.
Owned prefill target only; CARGO_INCREMENTAL=0, CPU test package debug0.
