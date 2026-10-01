# E154 independent final review

`independent-final-review.py --check` passes; `valid` and `adoption_ready` are true. The paired median FFN change is -2.424948% against E151. These are three independent contemporary allocations of the same frozen payload; their exact controls remain in the record.

| Job | Before | Candidate | After | Paired change |
| --- | ---: | ---: | ---: | ---: |
| 141633 | 295176 | 288044 | 295622 | -2.489853% |
| 141648 | 294688 | 288616 | 295855 | -2.254027% |
| 141661 | 294564 | 287581 | 294892 | -2.424948% |

All46 compiled inputs are bound. Exactly ops.rs, rmsnorm.rs and mlp.rs differ from E151; QKV/O images remain byte identical. The frozen E153 source composition plus reproduced E154 patch preserves original RMS through F32 DM, emits local BF16H480, computes the same finite Max on8valid shards (actual ratio256), produces8scale replicas and loads SubVRF directly. Scalar restoration to512consumers,23040B late packed replication, bounded480B term copies, ring32 and native/down arithmetic are verified separately from performance.

Nine distinct standard inputs pass (three repeated for performance, six extra). Actual new up/gate quantization passes8original cases/245760 BF16 values, allowing signed-zero equivalence. The original full-FFN construction passes3840exact values on the actual production image. Raw and whole sources, original fixtures, diagnostic/production image distinction, uploads and logs are checked by their frozen parsers.

The raw oracle excludes preRMS. Its RMS boundary is supported by source/emitted-prefix and coordinate checks plus complete-FFN tests; no intermediate RMS NPU value capture or arbitrary-input/model proof is claimed. Finite65280-pattern CPU Max/scale evidence has its own scope. No ranking or exclusive physical-bottleneck claim follows from this validation. Root owns live adoption and official submission.
