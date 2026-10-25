# Fixed unscaled E4 sliding KV: bounded archive and SDK audit

Read-only investigation; no cache/host/device implementation or compilation in
this lane. Root owns the prototype in `b16-fixed-e4-kv`. This records archive
facts and source review, not a performance or task-quality result.

## Distinct mechanism in the inspected archives

The original R64 FP8 reader, head-major Scale8 reader, B16 E4 reader, sole-E4
storage port, BF16-roundtrip quality hooks and later native-QK/headquant variants
all retain per-token/head power2 scales. `ops_kv_e4::quantize` computes Ds256
amax, derives a normal power2 scale and divides values before E4 casting.
`ops_kv_e4_pool::capture_b16/capture_p128` also reconstruct BF16 headed tensors,
writes E4 codes and repeats the scalar into Scale8. The old reader gathers both
codes and Scale8, widens codes through BF16/F32 and multiplies the scale before
materializing BF16. Native-E4 QK additionally quantizes Q and still reads KV
scales; it is not this fixed-cache/BF16-arithmetic proposal.

No already-tested fixed-unscaled KV route was found in these source/evidence
branches or the scoped all-ref commit-message search. This is a bounded archive
finding, not a proof that no untracked historical snippet ever existed.
Source hashes and inspected paths are in `source-facts.json`.

Prior HOLD was performance: all six public quality suites passed for the old
scaled candidate. Old real512-history warm C4 model53.304→53.896ms regressed,
C64 long105.526→100.619ms improved, C64 reset-short94.592→96.021ms regressed.
The old prompt32768 fill increased4.37%; later direct P128 readers partly reduced
that overhead, but c030 same-C3 HTTP remained slower. These are not current
B16 width16/context1024 measurements. See d51f4a73 audit and kv-profile's
`experiments/kv-e4-profile/actual-attribution.json` for retained provenance.

The new fixed scale1 proposal removes amax, exponent/scale derivation, DivF,
Scale8 writes/metadata/gathers, scale VRF loads and reconstruction multiply.
It also avoids mutating headed BF16 scratch merely to implement a scaled
roundtrip. It introduces finite-range/underflow loss and still requires an
E4-to-BF16 decode before the original BF16 QK/PV arithmetic.

## SDK0.9 legality and correction

Installed `src/cast.rs:27-42` permits E4→F32 but NOT direct
`fetch_cast::<bf16>()` from E4. `TableLookupCast<bf16>` at80-85 gives exact E4
widening. `CastEngineCast<f8e4m3> for f32` at145-147 supports append conversion:
BF16 fetch→F32→cast E4→commit/scatter. These type traits alone do not establish
engine placement.

Crucial restriction: `engine/fetch_adapter.rs:193-215` implements table lookup
ONLY for Tu::Main. My earlier suggestion of Sub LUT directly into K TRF was
incorrect and is withdrawn. Root's e0fbe296/r1262 failed with that exact type
error; keep the failed source/log. The existing O weight-stationary precedent
actually uses Main LUT→BF16 DM followed by plain Sub→TRF, not fused Sub LUT.

Supported causal repair: K E4 DM→Main LUT→BF16 DM→Sub collect/TRF. V already
uses Main for its transpose, so E4 LUT can precede the original BF16 transpose
without a separate scale/vector pass. Resource compilation must count K's new
materialization, LUT table loads, actual liveness and contention on Main; this
is not a free/fused K decode. A second fetch-cast stage or private SDK bypass is
not proposed.

The SDK host conversion (`float/f8e4.rs:14-60`) saturates finite overflow to
signed448, preserves signed zero and rounds magnitudes at/below2^-10 to zero
(tie at2^-10 rounds to zero); min nonzero is2^-9. NaNs map to the E4 NaN code.
Device conversion edge parity still needs an independent raw-byte gate, not an
assumption based only on the functional model. Actual K/V magnitudes and losses
must be measured; prior adaptive-scale quality cannot establish fixed-scale
quality. Keeping BF16 contractions does not make cache quantization exact.

## Complete-chain bound and memory

Current source distributes Ns8×Ts32 keys without GQA replication. Full B16
history payload is128MiB BF16; scaled E4 is64MiB codes+8MiB Scale8; fixed E4 is
64MiB. Live16 append writes64KiB E4 total for K+V versus128KiB BF16. Source
payloads are not measured hardware byte counters.

The matched d51f4a73/r930 trace has Task456950 raw cycles and four sequential
K/V gathers totaling227732 (49.84%). If only gather durations halved and all
other work stayed fixed, the gross saving would113866 cycles,24.92% of Task;
this is an optimistic proportional-service bound, not a prediction. New append
cast, K Main decode, LUT reloads and altered overlap reduce it. K→TRF was
already mostly overlapped, so its raw span cannot be counted as an additional
saving. O and BF16 softmax/PV/reduction remain unchanged. Compare the complete
current commit+attention+O graph before any host port.

With40 sliding layers, physical32 pools use10GiB BF16,5GiB fixed E4 or5.625GiB
old scaled E4. Physical64 doubles those numbers. An auxiliary mirror would add
5GiB at32 slots instead of replacing the BF16 pool; it is a different ownership
and peak-memory design and is not implemented/assumed here.

## e0fbe296 source review beyond the known Sub failure

Commit retains original VALID-only headed-source tiles and scatter addresses.
The caller must provide E4 byte offsets: token stride2048B and ring stride
2097152B, half BF16. No padded rows should be scattered. K layout remains
[B%4,Ts%32,Ds]; V's Ds8#32 E4 fetch widens into the original Ds8#16 BF16 collect
and unchanged transpose axes. QK/PV reductions, masks/softmax and Hsplit O are
unchanged by the diff. No owner alias/future-write change was found. Actual V
padding/order and all untouched pool bytes still require the hardware oracle.

Decode must encode/scatter current K/V before reading its ring, as today.
P128 uses immutable old history before committing its current128 rows; a future
port must preserve that chronology. If P128 current-chunk attention keeps its
original BF16 headed scratch, the quality control must reflect that exact
current-versus-history policy rather than silently applying the old scaled
roundtrip before current-chunk attention. All scalar/B16/R32/P128 producers,
reset/repack and corresponding readers must agree on the cache representation;
no reader-only host activation is valid.

Decision: a distinct, public-API mechanism worth one complete resource gate
using the supported K repair. Do not repeat old scaled-cache performance claims,
infer a50% complete speedup, or proceed to serving without actual codec/cache,
full-model and fresh public/private quality gates. This audit creates no code.
