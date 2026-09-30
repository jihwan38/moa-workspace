# D-E145 — E142 / E144 QKV readiness

Job **136924** completed the single bounded diagnostic with **81 comparisons PASS**. It observes the currently adopted E144 image: all six run/cluster samples start first V native **TRF-end+1**, and Q/K native **own-weight-end+2**. Thus both first-input readiness and later full-weight completion remain relevant. The result supports selecting a concrete replacement of one producer/consumer boundary; it does not measure a pure bus-stall fraction or establish a predicted saving.

| E144 landmark | Begin median | End median |
|---|---:|---:|
| preRMS apply | 8,025 | 8,988 |
| BF16 seed DMA | 8,996 | 13,424 |
| input ring32 | 13,426 | 17,637 |
| global K4 amax | 18,494 | 18,905 |
| power-of-two scale | 18,906 | 19,313 |
| quant-scale DMA | 22,113 | 28,985.5 |
| quant-scale VRF | 28,987.5 | 29,329.5 |
| first term | 29,330.5 | 30,180.5 |
| last term | 40,991.5 | 42,563.5 |
| activation TRF | 42,564.5 | 43,856.5 |
| V weight | 13,781 | 28,341.5 |
| V native + Switch + Term/K4 reduction | 43,857.5 | 48,932.5 |
| Q weight | 29,336.5 | 58,219.5 |
| Q native + Switch + Term/K4 reduction | 58,221.5 | 67,132.5 |
| K weight | 67,820.5 | 82,513.5 |
| K native + Switch + Term/K4 reduction | 82,515.5 | 87,589.5 |
| Q output | 82,522.5 | 83,076 |
| V cache write, cluster0 | 85,109 | 86,734 |
| K cache write, cluster0 | 97,386 | 98,986 |

These are medians of endpoints relative to each cluster Task. They must not be subtracted as if they were paired interval medians. Computing intervals **inside each run/cluster first** gives median preRMS-end→TRF-ready **34,300**, scale-ready→quant-scale-VRF-ready **9,914**, and last-native-end→Task-end **11,128 cycles**. Quant-scale DMA begins while V weight DMA is still active (same-observation median difference **−5,976 cycles**); its envelope includes overlapping readiness/waits, not isolated transfer service. Native markers include Switch, native contraction and Term/K4 reduction and cannot separate those costs.

The original whole standard hosts measured E142 **103,862 / 103,125**, bracket **103,493.5**, and E144 **95,706 cycles** (−7.524627%). This is one diagnostic allocation, not new independent adoption validation. Diagnostic info/trace medians were E142 **103,302 / 105,163** and E144 **95,574 / 98,994**. Instrumented trace timings are distinct from whole performance. O/FFN source and images are identical on both sides; their timing variation is not a QKV improvement.

Actual start commit is `08d9bb1597456c7fe7413229358d6d5077cdd7ba`. [Plan](plan.json) fixes E142 control `b3fe0ab…` and adopted E144 `3586c417…`. Both 46-file source manifests, all three original standard images, fixtures and runner are checked. The E142 diagnostic host is reused from D-E143 only after exact source/host-transform/image matching; E144 required one host build. Its QKV image is **206,592B / `687f1c8df3ccf24eee1e9425a64c685148620e09c4a245d4b039231380f3bc53`**. Existing marker-name bytes alone are changed. Instructions, tasks, addresses, constants, IO, marker counts and boundary IDs remain byte-identical.

[Independent review](independent-review.json) verifies ordered76/92 operation markers and E144's actual cluster0-only cos269, sin281, Vcache422 and Kcache500 descriptors, including public IO ancestry. Other absent markers are rejected. The historical control `graph.json` is preserved compiler **binary IR**, not a JSON graph; candidate graph is actual E144 compiler JSON. No E138 timing, source IDs or correctness verdict is inherited as E144 evidence.

[Full analysis](profile-analysis.json) binds every marker to its actual source/schedule and keeps all six observations. [Compact summary](readiness-summary.json), [actual-upload verification](verification.json) and [decision](decision.json) preserve scopes. Raw log and downloaded tar digests are in the screen result; the archive contains the exact four uploaded hosts and original inputs.

Three decisions are separate: keep adopted E144 unchanged; stop this fixed diagnostic and parameter tuning; let root select the next supported complete path from the current readiness evidence. Explicit weight staging must include extra DMA/native/assembly work; changing scale routing must account for input ownership and register lifetime. Neither proposed route is implemented here, and no overlapping intervals are summed as savings.

Reproduce the read-only checks:

```bash
python3 moa-dev/experiments/candidates/e144-qkv-readiness/independent-review.py --check
python3 moa-dev/experiments/candidates/e144-qkv-readiness/analyze.py e145-qkv-readiness-v1 136924 --check
python3 moa-dev/experiments/candidates/e144-qkv-readiness/verify.py e145-qkv-readiness-v1 136924 --check
```
