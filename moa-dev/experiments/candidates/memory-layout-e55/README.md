# E55 memory-layout experiments: E57–E64

**Retain E55.** Eight candidates passed all 15 official-harness comparisons per
Arena job, but none demonstrated a repeatable performance improvement. E59 and
E63 were measured in three independent jobs each; the E55 control was also
repeated three times. Fifteen unique jobs / 225 comparisons passed in total.

The [audit and results](../../../../docs/research/e55-memory-layout-audit.md)
trace the real model loader, all three kernel entry points, tensor layouts,
RNGD alignment rules, compiler descriptors, and the measured tradeoffs.
[hardware-summary.json](hardware-summary.json) verifies the downloaded upload
against each exact build and links every measurement.

| Variant | Result |
|---|---|
| `up-gate-packed` | E57: packed row pitch 1920→2048 B; FFN regression |
| `down-packed-split` | E58: packed row pitch 960→1024 B; FFN regression |
| `down-scale-split` | E59: scale row pitch 120→128 B; −0.24% overlaps repeat variation |
| `down-both-split` | E60: both down padding changes; FFN regression |
| `up-gate-bf16-skew32` | E61: BF16 row pitch +32 B; static gain, hardware regression |
| `down-bf16-skew32` | E62: BF16 row pitch +32 B; no measured gain |
| `qkv-sixteen-rows` | E63: E55 BF16 QKV with sixteen-way row interleave; no gain |
| `output-column-major-staged` | E64: different O load order plus required DM reorder; regression |

The six other directories preserve compile failures: three unsplit down
padding attempts, two 8-byte BF16 skew attempts, and direct column-major O.
Their `compile-failure.log` and `build-evidence.json` are not NPU results.

Each variant contains its complete changed source, a patch against the immutable
`baseline-e55` tag, and all 46 input hashes. Only `mlp.rs` or `projection.rs`
changes; official harness, fixture, public signatures and precision remain
unchanged. Each successful package also verifies that its other two device
images are byte-identical to E55. Unused padding is never treated as zero.

Reproduce from the workspace root with the existing Docker development container:

```bash
python3 moa-dev/experiments/candidates/memory-layout-e55/prepare.py down-scale-split
python3 moa-dev/experiments/candidates/memory-layout-e55/build.py down-scale-split
python3 moa-dev/experiments/candidates/memory-layout-e55/verify.py
python3 moa-dev/experiments/candidates/memory-layout-e55/analyze.py
```

`prepare.py` reads the tag and pinned Cargo lock without editing the runtime.
`build.py` compiles in `/tmp/moa-memory-layout-e55` in Docker, checks the actual
compiler inputs, builds the official host, verifies the embedded images, and
stages the frozen fixture and runner. The scripts need the historical local
artifacts referenced in the audit; large binaries are not stored in Git.
`verify.py` checks all checked-in variants. `analyze.py` rebuilds the complete
schedule and hardware summaries from those artifacts and tracked Arena reports.

One intermediate shared-target host was rejected before upload because it
embedded the previous candidate. `down-scale-split/host-cache-rejected.log`
preserves that log. The corrected builder removes only the experiment crate's
release units before the host build; SDK dependencies remain cached. Every
submitted host matches its exact compiled candidate image.
