# Actual public I32-owner binding result

Job **323014 SUCCEEDED in9.142186017s**. The packed-I32 input can be bound through
the checked public runtime API directly to the validated I4 projection image.
No device input reinterpretation or copy was added. **DOWN's resource screen
remains HOLD245,684>240,000**; successful binding does not waive it.

The captured/submitted ELF SHA256 is
`1cb160656f2cc0c1006184f6d7fafde0d61a3f7a49e2659cd3cee72eb5f69e5b`.
Build source4a13322d contains functional changece398503; submitted evidence is
cd331f3e. All404 build/submit input hashes agree, including original Cargo.lock
`a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e`.
The four projection images and the complete kernel section are byte-identical
to the actual-passed c0835bbf… projection ELF. This is a host binding change,
not a new matrix arithmetic result. Root submitted once with timeout70s and
without the profile environment; no additional Arena execution was performed
for this analysis.

## Actual checks

- **35,389,440 raw-versus-typed BF16 output comparisons PASS**.
- **17,694,720 raw row-permutation output comparisons PASS**.
- **52 negative-owner validations PASS**: (12 UP/GATE +14 DOWN) per round,
  two rounds. Wrong counts, each short slot, wrong ordering, unaligned input
  and overlap with every immutable slot are rejected before function lookup
  or launch. Poison outputs remain untouched before any valid call.
- The unchanged intended-E4 reference gate also passes35,389,440 BF16 outputs,
  3,072 sampled i64/F64 checks, and17,694,720 typed permutation comparisons.
- **571,922,440 immutable device bytes compared**, including9,830,400 additional
  bytes for the independently encoded I32 input owners across both rounds.
  These are repeated comparisons, not unique allocations. Source fingerprints
  and all post-reuse outputs are also unchanged.

The raw/typed/reference output sets overlap and are not added as unique model
outputs. The reference validates the specified quantized arithmetic; it is not
original-BF16 model accuracy. The exact local-dot and bounded F64 checks retain
the earlier preregistered numerical envelope. Address overflow has an explicit
checked guard, but no synthetic overflowing Buffer/address was constructed.
This actual run does not add a cancellation stress test.

## Matched ABI timing

Two fixed rounds, ABBA blocks of8 completed calls,32 timed calls per arm and
projection. All valid adapter size/alignment/overlap checks, public function
lookup and awaited runtime execution are inside the raw timer. All output
stores are included. Negative-case setup, CPU encoding, allocation, transfer,
readback, device quantization/packing and GeGLU are excluded.

| Projection | Raw-bound mean, µs | Typed-I4 mean, µs | Observed difference |
| --- | ---: | ---: | ---: |
| Joined UP/GATE |367.715500|367.598938|+0.116563µs (+0.03171%)|
| DOWN |263.546969|260.835156|+2.711813µs (+1.03967%)|

These are single-job block averages, not a zero-overhead guarantee or serving
throughput estimate. The separate typed-I4-versus-denseE4 ABBA is retained in
JSON and remains a comparison against the intended arithmetic reference with
an extra unity row-scale multiply. It is not used as the ABI-overhead control.

Separate measured preparation costs:

- CPU provided-weight reconstruction/encoding/audit:6,356.748308ms.
- Weight upload:230.522043ms.
- Activation fixture/CPU quantize/pack:266.156130ms and255.752344ms.
- Independent little-endian I32-word encoding and full byte audit:9.116205ms
  and6.950123ms,4,915,200B per round.
- Input upload/output poison/**negative validation and poison readback**:
  164.756963ms and67.450552ms. This label deliberately includes those checks;
  it is not a pure transfer measurement.

The392,129,540B live-payload figure is static accounting, not observed peak
allocator memory/RSS. No profile trace was enabled or collected in this job.

## Replay only

```bash
python3 experiments/int4-group960-bound/analyze_actual.py
```

The reader verifies preserved raw hashes, source/input identity, unchanged
kernel images, counts and ABBA scope, then prints
[actual-result.json](actual-result.json). `--verify-local-artifacts` also reads
and verifies the original current404 inputs and frozen/submitted ELF copies.
The reader performs no device or network operation. Raw inputs are preserved
under [actual-323014](actual-323014/raw-files.json).

This closes only the public packed-owner binding gate. A complete device
quantize→pack→consume path, nonlinear stages/fullFFN, model accuracy and serving
performance remain separate missing evidence. No model activation is included.
