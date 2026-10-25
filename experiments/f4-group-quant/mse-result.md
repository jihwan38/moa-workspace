# Fitted dyadic scales: CPU proof PASS; provided-weight gate ready

Source `8e7ce99e` implements the fixed25-choice plan in `mse-plan.md`. It uses
exact u64 histogram SSE, keeps the original ceil-max candidate on ties, retains
signed zero, and rejects either E4 NaN. No device kernel or serving route is
changed. Independent source reviews found no scale/index/overflow defect.

The compiled helper passed an independent Fraction oracle for120,650 finite
candidate-code decisions,475 scale bit patterns,950 rejected nonfinite choices,
1,843,200 packed fixture values,1,920 selected group scales and5,760 reported
error/norm sums. The fixtures include true row/K permutations, zero, subnormal,
extreme and concentrated/outlier histograms. Every selected group has SSE no
greater than the original candidate; this is a current-E4 weight-space property,
not a full-model or task-quality guarantee. All zero groups keep scale1.

CPU-only one-thread encode benchmark on the same59M-value fixture measured
79.59–90.12ms across five calls, versus12.73–17.52ms for the earlier packed-only
ceil-max helper. Fitting includes the initial immutable LUT construction in the
first call. Observed process VmHWM was89,444KiB, including source and packed
output. No transfers, model startup or GPU/CPU error-audit pass are timed. The
fitting cost must be included when considering any model startup integration.

`test_f4_mse_weights` is a supplied-weight-only diagnostic: it loads the actual
current native E4 weights, processes all12 projections from layers0/5/6/47 one
at a time, reports old/new L2 and row-dequant-weighted L2, and checks64 complete
groups per projection with independent F64 decoders. Its all-group error reports
reuse the already computed exact histograms, avoiding a full per-value F64 audit.
D2H, old encoding, and fitted encoding/report timing are labeled separately.
It uploads no F4 weights and invokes no F4 consumer or transformer forward pass.

Host check passed10.65s; recorded release build passed36.18s, with all source
hashes unchanged. Frozen artifact:

```
experiments/f4-group-quant/artifacts/test_f4_mse_weights-8e7ce99e
SHA256 14126077b76bbcdb78d72ab309a8dee5ef199030bace33b344cb969385922294
4,425,584 bytes
```

Default arguments select all four layers; `0 2` and `2 4` split the same fixed
cases. Root owns NPU dispatch. No supplied-weight result has been inferred from
the synthetic CPU proof or compiler success.
