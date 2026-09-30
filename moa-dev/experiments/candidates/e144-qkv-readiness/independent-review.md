# Independent D-E145 pre-runtime gate

`independent-review.py --check` passes. Both diagnostic hosts contain the actual E142/E144 images, with only profile-name bytes replaced in QKV. Original task bytes, addresses, I/O, constants and the O/FFN images are unchanged. The source manifests contain46 inputs each; the diagnostic change is restricted to the host harness. Image metadata and schedule operation kinds agree for all76 control and92 candidate markers in increasing marker order.

Actual E144 graph descriptors identify schedule269 as cosine gather (IO12),281 as sine gather (IO13),422 as V-cache scatter (IO15) and500 as K-cache scatter (IO14). All four descriptors execute only on chip0/cluster0/PE0, so their absence on cluster1 is expected. Other missing markers must not be excused by this whitelist. The preserved E142 `graph.json` is binary IR; exact E142 image identity binds its schedule, but this review does not decode that control IR.

This verifies the measurement boundary, not runtime timings or a bottleneck. Analyze each run/cluster clock separately; retain plain-whole/info/trace costs and do not add overlapping intervals. E138-v2's prior weight-readiness observations are not E144 measurements. No compile, NPU launch or production mutation was performed by this reviewer.
