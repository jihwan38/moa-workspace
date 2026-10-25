# B16 per960 F4 partial-scale: actual gate

The complete candidate passes its intended arithmetic reference and clears the
predeclared 10% speed gate. It is **not equivalent to current E4 weights**, and
there is no whole-model quality or serving result. Keep production unchanged
until startup, capacity, and model-quality gates resolve those separate risks.

Root dispatched the unchanged ELF `ab63a46d…3865d`, built from `ed26f823`, as
r1204/job321190 (provided layers0/5, 15.705s) and r1205/job321191 (layers6/47,
19.625s). Both succeeded. Full source/build provenance is in `build.json`;
`analyze.py` regenerates `actual-r1204-r1205.json` from the retained logs.

## Correctness scope

- 18,679,808 retained stage bytes match the independent dense E4 expansion of
  the **same unscaled F4 codes**, with the same per960 F32 partial scales.
  This covers all eight buffers, both row permutations, and two feedback steps.
- 1,773,066,352 original/packed/dense weight, scale, and norm bytes remain
  immutable. Separately poisoned candidate buffers are consumed first.
- 1,536 independently decoded F64 local-dot samples and256 final DOWN samples
  pass; observed error is zero for every sample. The predeclared checks are
  toleranced, so this is not a CPU all-output bit-equivalence claim.
- Normalized BF16 input, E4 input codes, and input scales match current E4
  exactly. Only the weight representation changes. Layer47 also uses a signed
  gate. The supplied-layer tensors are real model weights; activation fixtures
  are synthetic, non-dyadic, distinct rows with zero and scaled rows.

At each feedback step, all three arms receive the same previous candidate
state. Reported original errors therefore compare one layer on matched input;
they do not measure independent original/candidate multi-layer trajectories.

| Comparison against current E4 | Relative L2 range |
| --- | ---: |
| Reconstructed F4 weights | 0.14087–0.18328 |
| Weights including retained row-dequant scale | 0.14177–0.19525 |
| BF16 DOWN projection | 0.15721–0.25812 |
| Final postnorm/residual/gate state | 0.16067–0.20353 |
| BF16 GeGLU hidden | 0.19054–0.26677 |

Raw E4 hidden-code comparisons also change (relative L2 0.23833–0.44123), but
that buffer's per-row scale changes too; code error alone is not an error metric
for reconstructed hidden values. These material differences require actual
model quality testing. The reference is current native E4, not original supplied
NVFP4; no original-NVFP4 error or token-selection result is claimed.

## Complete timing and actual trace

Each timed sample resets identical projected/residual inputs before its timer,
then awaits one complete postattention/norm/UP+GATE/GeGLU/hidden-quant/DOWN/
postnorm call. All lookup, group-scale, reduced-F32 DM, and retained HBM costs
are inside. Encoding, upload, snapshots, and immutable readbacks are outside.
Every reported ABBA block averages eight completed calls; both permutations
use A/B/B/A. There are128 timed calls per arm across the four layers.

| Layer | Current E4 mean µs | F4 mean µs | Reduction |
| --- | ---: | ---: | ---: |
| 0 | 258.529 | 196.877 | 23.85% |
| 5 | 257.652 | 196.765 | 23.63% |
| 6 | 258.282 | 197.218 | 23.64% |
| 47 | 259.860 | 197.094 | 24.15% |
| Aggregate | 258.581 | 196.988 | 23.82% |

ABBA block ranges are255.503–261.832µs current and193.507–199.517µs candidate.
Separate actual profiling was enabled through `FURIOSA_OPT_PROFILE=trace`.
Across the four layers/two clusters:

- Task spans:487,774–493,691→371,376–375,854 cycles, mean490,937→373,936.
- UP weight DMA spans:96,320–100,498→46,800–48,306 cycles.
- GATE weight DMA spans:95,290–100,322→47,611–48,195 cycles.
- DOWN weight DMA spans:95,590–99,508→48,772–49,470 cycles.
- Per-cluster DMA count36→42; TU count46→49. Mean summed TU spans
  126,792→178,294 cycles; three new table-load spans total about17,309 cycles.

Weight DMA identity is matched to the frozen resource's DMA order, including
both loop iterations, and the reader verifies36/42 actual DMA counts. DMA and
TU sums overlap and can include queued waiting; do not add them or call them
exclusive device busy time. The actual Task and completed-call reductions are
the end-to-end evidence. Half-sized weight reads overcome the extra lookup/
scale work here; this is not a claim that native I4 doubles arithmetic throughput.

## Static encoding is a separate gate

The existing frozen helper already uses a19×256 **integer lookup table**.
`encode_finite` searches eight exact integer distances only while constructing
that tiny table. Each model value then uses `table[byte]`; there is no
per-weight F64 nearest search to remove.

`Projection::new` starts `cpu_encode_and_audit_ms` after reading the original E4
tensor, calls the helper, then independently decodes and audits every one of
58,982,400 values using F64. It stops before any new HBM upload. The logged
1.546–2.266s/projection is **encode plus exhaustive audit**, not encoder-only
time, total startup time, or an estimate for48 layers. Likewise, the15–20s jobs
include weight loading, reference materialization, correctness readbacks,
partial-oracle kernels, and timing/profiling.

The next bounded plan is an encode-only CPU/RSS benchmark on one full provided
projection, with the current helper retained as the byte reference. Measure
any production-only packed/scales encoder separately from the dense oracle
output and F64 audit. Validate every packed byte/group scale, signed zero,
threshold tie, group maximum, and invalid-input rejection. Then separately
measure the ordinary source readback and packed/scales upload if conversion is
performed after current device prepack. A single projected48-layer estimate
must state its assumptions and cannot replace actual startup measurement.

Declared size for one projection:58,982,400 E4 source bytes→29,491,200 F4 bytes
+245,760 F32 group-scale bytes. All three projections across48 layers total
4,246,732,800 packed bytes+35,389,440 scale bytes=4,282,122,240 bytes, excluding
unchanged row/global scales. Keeping these alongside E4 controls adds this
storage; it does not automatically replace existing joined FFN allocations.
The dense E4 oracle copy is diagnostic-only and need not be resident in a future
model candidate. No48-layer allocation, CPU conversion, host dispatch, or cache
port is authorized by this micro result alone.
