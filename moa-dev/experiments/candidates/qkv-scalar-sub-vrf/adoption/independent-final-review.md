# E147 independent final review

The frozen [offline verifier](independent-final-review.py) and [result](independent-final-review.json) report `valid: true` and `adoption_ready: true`. Root owns the final adoption decision and live source changes. Reproduce with `python3 adoption/independent-final-review.py --check`.

The actual 46-file build changes only `projection.rs` from E144. QKV uses the measured E147 image; O and FFN images remain exactly E144. The finite-BF16 scale proof, complete emitted scalar/bitmap/VRF gate, original host fixtures, actual uploads, image manifests and raw logs are bound to the result. The original host gate-wait timeout is retained; package resumption reused the completed host and identical images without recompilation.

Three independent contemporary brackets improve QKV by 1.162207%, 2.262397% and 0.796883%. The paired median improvement is 1.162207%; all three candidates are faster than both corresponding controls. This finite sample is not a statistical significance or ranking claim.

Nine distinct standard inputs pass: the original three were repeated across the performance allocations and six additional inputs were tested separately. The unchanged raw nine-case oracle passes 73,728 first-BF16 outputs exactly (signed zero equivalent), with 63 actual input hashes. The two original complete-QKV boundary cases pass 16,384 outputs and preserve 8,380,416 untouched cache entries, with 38 actual input hashes and the original public tolerances.

The separately compiled raw graph was checked independently: its one Main scale producer, two direct Sub-to-VRF ring4 broadcasts, shared bitmap producer, three full-H weight/native paths and register lifetimes match the complete image's contracts. Graph-local bitmap tensor identifiers differ and are normalized only after verifying both actual bitmap producer descriptors. All 11 raw VRF allocations have valid, nonoverlapping live ranges. The separate diagnostic image is not misidentified as the full-QKV image.

The scale identity is exhaustive over 65,280 finite BF16 encodings and relies on monotonic max composition. This does not establish arbitrary-input native-QKV accuracy, nonfinite behavior, end-to-end model correctness, or physical bottleneck attribution. New per-input multiplication and Sub Switch/configuration work are included in the whole measured path; removed DM traffic is not treated as free time.
