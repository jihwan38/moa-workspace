# E140 copy-only correction: adoption review

The frozen `control-copy-fix/mlp.rs` is E137 with only three up-quantizer copies repaired. Each source owns480 initialized bytes; the old typed DMA emitted1440B reads and1440B writes at packed offsets0/480/960. The corrected Sub streams copy15×32B per term and exactly cover the1440B packed output. Literal source reconstruction and actual emitted descriptors verify this contract. Weighted down, arithmetic, QKV and O remain unchanged.

Three independent, contemporary E137 → copy-fix → E137 jobs111461/111475/111477 passed all135 comparisons, including45 candidate comparisons. Paired FFN changes are−0.210119%,−0.184712%,−0.056763%. These small differences support neutral performance, not a statistically resolved speedup. The memory correction is warranted independently of those differences.

Extra6 job111494 passed30 more candidate comparisons on original standard input indices3..8, with byte-identical production images. Combined coverage is nine distinct standard cases; repeats do not add distinct inputs. Whole-FFN regression111488 also passed3840/3840 exact BF16 outputs using the same13 tensor inputs and official PyTorch reference as the E134 precision regression. Uploaded payloads, raw logs and actual image/source bindings are rechecked. The passing local-down raw9 oracle is not inherited: it excludes the modified up prefix and final global-scale application.

`final-verification.json` binds46 compiler inputs, the exact FFN image, the unchanged E137 QKV/O images, these measurements and the full-FFN regression. `adoption-review.json` records the three decisions: root reviews adoption as a correctness repair, parameter tuning stops, and the local-down architecture remains unadopted after its separate3.098140% regression. Historical first-screen and failed local-v4 evidence remain immutable. No runtime, baseline or official submission changes occur in this directory.

Recheck from repository root:

```sh
python3 moa-dev/experiments/candidates/down-native-consumer-local/control_validation_verify.py --check
python3 moa-dev/experiments/candidates/down-native-consumer-local/control_provenance.py --check
```

The first command rechecks the three uploaded performance batches, extra6 archive/log, exact source/descriptor gate, source/package provenance and original full-FFN stress verifier. It does not compile or launch an NPU job. Finite tests do not prove arbitrary-input or end-to-end model correctness. `measured-build-e140-copy-adoption.json`, `compiled-copy-source-manifest.json` and `control-source-build-binding.json` are the source inputs for root's later source-only adoption/tag and baseline metadata.
