# Conditional repair of E138 input routing

This note records an implementation candidate; native-v1 remains frozen until the one authorized NPU profile selects it. Whole111168 passes45 comparisons and regresses12.900619%. The static12,328-cycle typed input transfer is not evidence of a physical stall or an amount of reclaimable time.

Current path is post-preRMS BF16 `[2]/[1#32,H/480]/[H%480]` → typed DM `[2]/[QkvGroup64,H/960]/[H%960]`. Each `(c,g)` has H3840, split into q<4 shards, and quantizes independently with the same global H3840 scale. The actual descriptor copies1024×960B, twice per H960 consumer, totaling983,040B.

A concrete alternative keeps every quantizer/native/scale/head operation after that input unchanged:

1. Typed DM to `[2]/[Dummy256/32,1#8,H/960]/[H%960]`, containing8 rings×4 column anchors per cluster. Each q anchor holds x[960q..960q+959]. The logical seed payload is122,880B.
2. Main BF16 `fetch<1,H%960>()` → `switch<[Dummy256/32,8,H/960],1>(CustomBroadcast{ring_size:32})` → `collect<H/16%60,H%16>()` → exact BF16 commit.
3. Full-allocation relabel from `[Dummy256/32,8,H/960]` to `[QkvGroup64,H/960]`. Physical slice32a+4b+q becomes4g+q with g=8a+b. No padding promotion or row/K transposition occurs; every group again has identical H3840.

`ffn-native-down-weighted-residual/weighted-v3/mlp.rs:262–275` already uses the identical four-column ring32 arrangement for E5 data. E137's original QKV call site uses BF16 CustomBroadcast ring32. The combined BF16/four-column expression still needs its own exact descriptor and whole-NPU proof; neither prior proves its complete cost here. It uses a real DM commit then plain Sub TRF, and does not invoke the unsupported direct Sub Switch→TRF shortcut.

The trade is983,040B direct typed transfer replaced by122,880B seed typed transfer, **plus** a Main Switch reading122,880 logical valid bytes and writing983,040B, with additional122,880B seed storage. The consumer quantizer remains active on512 slices, its1.875MiB four-term DM and shared TRF remain, and original30MiB weight HBM and F32 head gathers remain. No factor/term/row/K parameter changes are involved. Packet-count and logical-byte changes must not be treated as measured bus savings or predicted speedup.

Select this repair only if actual input distribution materially delays activation/native or competing weight issue. If weights or later heads determine completion with enough activation slack, this note does not justify implementation. If selected: freeze a new variant and recipe, verify all128 group inputs and unchanged math by actual descriptors, run early whole QKV, and rebind both root-owned raw9 and full2 suites to the exact new image. Positive whole gain, including below5%, leads to bounded adoption validation.
