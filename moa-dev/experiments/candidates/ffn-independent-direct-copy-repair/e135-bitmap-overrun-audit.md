# E135: up-quantizer overflow reaches a live Switch control bitmap

The unnamed32B/slice buffer is **the activation ring32 `custom_snoop_bitmap`**, not an FP4 lookup table. Whole graph O43 loads T58 from embedded `Weight+45056`; O44 explicitly uses T58 as `custom_snoop_bitmap` for `CustomWithBitmap{ring_size:32}`. Schedule tensor35 is its allocation and instruction273 consumes it in `broadcast_ffn_quantized` (`mlp.rs:581`).

The embedded exact image contains the expected16,384-byte bitmap exactly once, at file offset45,103: for both clusters and each destination slice `d`, its256-bit little-endian mask has the single bit `32*floor(d/32)`. This independently matches the emitted graph's256 `SliceBitmap` entries. The image hash matches the original exact-build record.

| Frozen image | Packed base/extent per slice | Live bitmap | Lo write | Third write | Bitmap load → consumer |
|---|---|---|---|---|---|
| Whole E135 |17,664/1,440B|[19,200,19,232)|[18,144,19,584)|[18,624,20,064)|2,600–3,319 →45,100–56,883|
| Original up/gate oracle |15,360/1,440B|[16,896,16,928)|[15,840,17,280)|[16,320,17,760)|1,550–2,269 →18,939–30,722|

Both images inherit compact480B hi/lo/third sources copied by1440B typedDMA descriptors. Their lo and third writes overlap all32 bytes of the bitmap on slices0..7 of both clusters: **16 control entries**. The bitmap was already loaded and remains live; the Switch reads it later. The standalone oracle's graphT41/scheduleT15 has the same role and embedded pattern, found uniquely at file offset12,335 (`Weight+12288`).

Original whole110314 and standalone oracle110316 both time out on their first candidate launch withcode4 and produce no candidate FFN/upgate arithmetic outputs. The graph and static lifetime evidence establish a common, concrete control-memory corruption hazard. They do **not** locate the actual timeout instruction, show corrupted runtime bits, prove a particular NoC mechanism, or prove the compiler/SDK is faulty.

A bounded revival of the **same E135 family** is justified: change only these three copies to exact480B Sub commits, confirm read/write bounds0..480 and destination offsets0/480/960, then test whole plus the same up/gate oracle. A successful repair would strengthen causal attribution; it is not guaranteed by this audit. Compose current E137 QKV/O with the frozen E135 FFN explicitly in the new source recipe, since old E135 was built from E134.

`moa-e135-bitmap-overrun-audit.py --root <workspace> [--output <json>] [--check]` reproduces the assertions without compiling or running the NPU. JSON binds source/build/graph/schedule/exact-image/raw-log hashes for both original failures. No frozen historical records were changed.
