# QKV normalize directly to ring anchors

Exact compile passed; static 56844 cycles versus 56,719 for replicated-input normalization plus gather plus ring32 anchors. NPU untested.

The added `normalize_anchors` preserves the existing eight-slice F32 reduction, EPS, sqrt, divide and weight multiplication. It casts each final normalized shard to BF16 before typed DMA into eight ring32 anchors per cluster. The old single-slice full-H gather and separate anchor copy are removed. Existing generic `normalize` is unchanged. Casting moves across copies only, with no intervening arithmetic.

`ops.rs` is a whole-file snapshot for reproducibility; apply only its incremental patch to avoid reverting unrelated experiments. Both-cluster input normalization and ring32 remain unchanged.

Schedule is under ignored `moa-dev/experiments/artifacts/qkv-normalize-anchors/schedule.json`; source and schedule hashes are in `evidence.json`. Compile used independent `/tmp/moa-qkv-normalize-anchors-target`. The first attempt failed because the mapping macro does not accept a qualified axis path; importing Dummy256 fixed the syntax.

Decision: do not adopt. Static makespan increases by125 cycles (0.22%); retain the hardware-tested normalization/gather path.
