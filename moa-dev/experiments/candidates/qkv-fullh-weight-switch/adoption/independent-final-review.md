# E144 independent adoption evidence review

**The frozen `e144-fullh-v2` evidence is valid and ready for root adoption review.** No source or runtime was adopted by this verifier. The machine-readable gate is [independent-final-review.json](independent-final-review.json); reproduce it with `python3 moa-dev/experiments/candidates/qkv-fullh-weight-switch/adoption/independent-final-review.py --check`.

Three independent Arena allocations used identical candidate/control payloads and contemporary E142 before/after controls. Recomputing QKV ratios directly from the original stage results gives:

| Job | E142 before / after | E144 | Paired change |
| --- | ---: | ---: | ---: |
| 136536 | 103820 / 103348 | 95909 | −7.409445% |
| 136554 | 104003 / 104574 | 95910 | −8.033963% |
| 136573 | 105330 / 103125 | 95899 | −7.990693% |

The median paired reduction is **7.990693%**. The ratio of aggregate medians, a different calculation, is7.981099%. All three allocations improve; repeated inputs and before/after stages are not counted as additional independent allocations. These observations do not establish a ranking outcome or a statistical confidence interval.

All required finite correctness checks passed:

- Original3 standard inputs and6 additional run IDs3–8, with all30 additional comparisons passing in job136585. The extra harness preserves the original PRNG/reference/tolerance and embeds the identical3 device images.
- Actual pre-head-RMS production helper:9 mathematical cases,73728 BF16 outputs exact with signed zero equivalence, and63 original input SHA checks in job136561. The wrapper, Rust host, F64 generator and truth match the original E138 oracle bytes. The separately compiled diagnostic graph was checked independently: all3 weight/native paths match the whole image's emitted descriptions and shapes, the3 native tasks share valid TRF/VRF lifetimes, and its9 VRF allocations have no overlapping live address ranges. This diagnostic image is not presented as the whole production image.
- Actual complete QKV image:16384 outputs in2 original cancellation cases and8380416 untouched cache entries pass in job136578, with38 original input hashes. This retains the original full2 tolerances `atol=.04`, `rtol=.01`; no acceptance condition was changed.

The gate binds the actual46 compiled source files and the206592B QKV image `687f1c8df3ccf24eee1e9425a64c685148620e09c4a245d4b039231380f3bc53`. Only `src/ops.rs` and `src/device/sliding/projection.rs` differ from E142. E142's MLP is preserved, and the O/FFN images are byte-identical to the control. The previously frozen actual-graph audit establishes32768 row-quarter correspondences, exact30MiB weight source coverage in256B packets, canonical native row restoration and unchanged remaining191 operators. All result logs and uploaded archives are rechecked through their frozen validators, then linked by digest here.

The passing prototype is v2's low-two-bit row ownership. V1's actual Switch output-time mismatch remains a preserved failure; it was repaired by matching the producer's row ownership to the supported Switch ordering, without adding a full weight transpose. The new Main Switch/setup/collect and changed DM reader costs remain part of every whole measurement. The result does not isolate a pure HBM-bandwidth gain, remove all quantization cost, or prove arbitrary-BF16/end-to-end model accuracy. No hidden DPE SFR or physical bank timing guarantee is inferred from the structural audit. Root owns the final adoption, live-source binding and official submission decision.
