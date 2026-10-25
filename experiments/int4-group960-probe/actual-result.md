# Actual layer0 intended-code projection result

Job **322975 SUCCEEDED in9.610338484s**. The actual numerical gate passed;
DOWN's original resource screen remains **HOLD245,684>240,000**. This report
neither activates a model path nor establishes model quality or complete FFN
speed.

The root submitted the captured ELF
`c0835bbf4fcd57cc2611b3c39deb566ec26a107efa157f2854d9420f161e1ab1` from
a003, with `FURIOSA_OPT_PROFILE=1` and timeout70s. Build functional source is
7b9aef54, build snapshot e72c4cb1, submitted evidence revision0e3588da.
All402 build/submit source/config hashes agree; original Cargo.lock remains
`a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e`.
All four newly compiled images match the images embedded in the captured ELF.
No additional device execution was performed for this analysis.

## Correctness scope

The two fixed rounds use all512 rows, then the actual row permutation
`(73*p+127)%512`, with provided layer0 UP/GATE/DOWN weights independently
reconstructed as BF16(providedF4*providedE4), before global scaling. CPU exact
rational codes are selected independently per960 weight group and per full
activation row. Both paths consume identical signed integer codes; the E4
reference represents every code exactly. It is an intended-algorithm reference,
not an original-model BF16 reference.

The actual assertions passed:

- **35,389,440 BF16 output comparisons**, all finite and bit-exact against the
  separate P64 denseE4 reference.
- **3,072 independent sampled i64/F64 checks**,512 per projection per round,
  under the preregistered rounding envelope. Local integer sums are exact;
  the F64 center allows its own summation rounding error. This is not a claim
  that every output equals an unrounded F64 calculation.
- **17,694,720 candidate permutation outputs**; the reference permutation is
  checked as well. These overlap the parity workload and are not added as a
  separate set of unique outputs.
- **562,092,040 immutable device-byte comparisons** after all repeated calls,
  plus unchanged source tensor fingerprints. Counts include checking the same
  weights across both rounds; they are not unique allocated bytes.

The exact weight-code and source FNV1a fingerprints are preserved in the raw
log and JSON. They are explicitly noncryptographic; SHA256 binds all raw files,
source inputs and the executable separately.

## Completed projection timings

Each of two rounds runs ABBA blocks of8 completed awaits per projection. Thus
there are32 timed calls per arm/projection. All actual output stores are inside
the timer. The table averages equally sized blocks; ranges are block means.

| Projection | I4 candidate range, µs | I4 mean, µs | DenseE4 intended mean, µs | Mean latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Joined UP/GATE |367.903875–371.223875|369.329531|507.090781|27.16698%|
| DOWN, including BF16→global→BF16 |260.586750–262.758000|261.745156|360.749094|27.44399%|

The independent reference retains a disclosed extra unity row-scale multiply.
These percentages are relative to that reference and cannot be treated as a
speedup over the current production P512 FFN. UP/GATE global scaling, GeGLU,
activation device quantization, device packing, allocation, upload and download
are outside these timers. Summing these projection means does not supply the
missing complete quantize/pack/consume measurement.

Separate observed preparation costs:

- CPU weight load/reconstruction/rational encoding/packing/audits:
  **7,224.436536ms**,176,947,200 weight values/184,320 scale groups.
- Weight upload: **229.513330ms**.
- CPU activation fixture/quantize/pack:268.019847ms and257.490669ms.
- Input upload/output NaN initialization:103.490072ms and107.960934ms.

These include the probe's independent checks and temporary buffers; they are
not a production48-layer startup measurement. The351,824,900B device-payload
figure is static accounting, not an allocator or process-RSS measurement.
Four untimed profile markers appear, but no per-task trace payload was collected
in the preserved `arena.log` or the empty `arena-records.json`. No DMA-span,
cycle-frequency or overlap attribution is inferred from those markers.

## Reproduce this analysis only

From the worktree root:

```bash
python3 experiments/int4-group960-probe/analyze_actual.py
```

The script reads the preserved raw files, verifies hashes and assertions, and
prints the JSON report without launching a device job. On the original machine,
`--verify-local-artifacts` additionally rechecks all402 current source/config
files and both original captured/submitted ELF copies. Output matches
[actual-result.json](actual-result.json). Raw inputs and their SHA256 manifest
are in [actual-322975](actual-322975/raw-files.json); initial hypotheses and
bounds remain in [plan.md](plan.md).

The next missing evidence would be a separately authorized complete path with
actual activation quantization, packing, nonlinear stages and all boundaries,
followed by independent model/task-quality evaluation. Neither is authorized
or established by this result. The DOWN resource HOLD is unchanged.
