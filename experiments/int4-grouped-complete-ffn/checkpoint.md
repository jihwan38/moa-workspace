# Source and plain hostcheck checkpoint

Plan `c20b82b2`, functional assembly `394d721e`, based on projection `71cdf450`.
`a001-int4-grouped-complete-ffn-hostcheck` passed in11.10s using the owned
prefill target, original Cargo.toml/lock and `cargo check --locked --bin
test_int4_grouped_complete_ffn`. Its command, timestamps, log and all input
hashes are retained in `a001-hostcheck/` and `hostcheck.json`. No resource
compile, standalone ELF or Arena execution was performed by this tree.

`audit_source.py` rechecks14 preserved files, the exact committed quantizer
donor bytes, original dependency lock, unchanged production/prefix calls and
the same six remaining awaits in the clocked path. All inherited `src` files
remain unchanged except the additive module export. Original complete323067
and drift323102 controls and outcomes remain intact. There is no new device
arithmetic in this integration.

The new probe checks196,608,000 intended BF16 comparison events,6144 independent
i64/F64 samples,40,960 grouped scale values,40,960 relative offsets and
19,660,800 packed bytes over two feedback steps/two row orders. All candidate
eleven retained owners and all production scratch are checked after ABBA,
phase-clock and optional profile reuse. These are planned checks, not results.

Root separately reports component actual gates323173 (quantizer) and323172
(projection) passed. Their standalone timings have different inputs/control
scopes and must not be summed or interpreted as this complete graph's speed.
The current complete production graph remains the only performance control;
the >=10% gain gate is unchanged and must pass in both row orders. A later
fresh ELF must bind all reused images, including the two new grouped quantizer
images and four grouped projection/reference images, to their component gates.

Zero fractions and scalar RN32 reconstruction error are reported on each arm's
own BF16/quantized state. Step0 output drift is matched-input; step1 is divergent
feedback. All activations remain synthetic and provided weights/norms are only
layer0. This source/check checkpoint establishes no full FFN speed, model
quality, all48-layer capacity, startup or serving claim.

## Fresh build follow-up

Root and round2 source review passed. Fresh captured build `a002` completed in
55.88s from `dc3524a3`; all421 input hashes and all19 prior actual device images
match, including the unchanged q020 production whole-graph control. The13 old
complete entries,2 grouped quantizers and4 grouped projection/reference entries
are individually recorded in `a002-image-audit.json`; `audit_build.py` replays
all comparisons without compiling or running a device. The earlier source-only
checkpoint above remains historical. Root alone owns the pending hardware run;
no complete speed or quality claim follows from this build.

## Actual complete result

Root job323213 passed all intended-code/ownership checks but failed both
preregistered10% whole-chain performance gates (about2.1% slower than current
production). `actual-result.json` and `analyze_actual.py` preserve every ABBA
and phase sample, independent counts, source/image binding and synthetic drift
scope. HOLD: no model integration, quality or serving promotion.
