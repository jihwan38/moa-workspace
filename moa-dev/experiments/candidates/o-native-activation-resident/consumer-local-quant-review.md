# E137 consumer-local four-term quantization: conditional layout review

Read-only design review of frozen native-v2 and installed SDK 0.8.1. No v3 source, compile, graph, NPU or speedup claim is created by this note. Four factors remain [1,16,2^22,2^27]. Proceed only if the frozen whole-kernel profile identifies global-max/packed-term redistribution on the limiting path and the parent selects this repair.

## Complete coordinate contract

Let c∈[0,2), g∈[0,16), q∈[0,16), u∈[0,256). Destination slice s=16g+q must contain the original BF16 HBM value x[256q+u]. HBM byte offsets are 2*(256q+u), always in [0,8192); each (c,g) must receive every q exactly once. The input source lacks c and Hgroup axes, so the existing public HBM to_dm/transpose-broadcast path is the relevant API; matching physical dimensions alone does not prove broadcast ownership.

For each (c,g), local max over u then inter-slice Max over the sixteen adjacent q slices gives the same full-4096 amax. All32 groups see bit-identical finite BF16 bytes. Absolute-value conversion, Max, the existing positive floor, exponent mask and multiplication by 2^-7 are deterministic here, so all32 scales are equal. This is not a cross-group reduction or a per-K scale approximation. It does not cover NaN behavior. The three residual reconstructions and fourth cast must retain the exact frozen arithmetic order.

The new max result layout is [Hgroup16, 1#16], with one valid q=0 scalar32B per group. Whole-allocation reshape to [anonymous16, 1#16] preserves every physical slice, q-validity mask and element byte. If cluster mapping is Dummy2, a relabel to anonymous2 requires the same proven cross-cluster equality. Do not shrink Slice to [1#256] or reinterpret padded q slots as initialized replicas. The unsafe relabel is justified by scalar equality as well as unchanged wire order: later replication may select any valid group because all are equal. Then ordinary to_dm can target canonical Cluster=[1#2], EpilogueSlices=[1#32,H/480], Element=[1#8]. Actual graph must show selected source c,g,q=0 within initialized storage and eight canonical destination slices, not a lingering Hgroup-to-H-output coordinate match.

Public SDK evidence: furiosa-opt-std-0.8.1/src/tensor/memory.rs HbmTensor::to_dm lines371–386 and DmTensor::to_dm lines955–965 call transpose(true) after alignment checks; DmTensor::reshape lines968–995 explicitly requires per-level sizes and identical physical wire order. tensor/mod.rs lines275–302 reiterates that reshape is a relabel, not a permutation. tensor/memory/dma_layout.rs checks destination tail/stride alignment but does not prove semantic equality of replicated values. Existing E134 broadcast_geglu_scale already relabels fully broadcast scale ownership, but that is evidence for the API pattern, not v3 lowering proof. Existing E137 native K16 reduction demonstrates the intended adjacent-q reduction geometry.

## Removed and added work

- Remove the packed-term redistribution: v2 writes32KiB at32 active shard owners and transfers512KiB into consumer allocations. v3 directly writes the consumer512KiB packed payload and uses it for the unchanged TRF load. The512KiB transfer disappears, but512KiB term materialization remains.
- Input replicated destination payload grows16KiB→256KiB, an additional240KiB. Logical descriptor payload is not measured HBM bus traffic; multicast/coalescing and scheduling require emitted descriptors/runtime evidence.
- Max and four quantization input passes now activate512 rather than32 slices. Their combined logical BF16 DM reads grow80KiB→1280KiB. The source program can issue the same instruction shapes across more slices; this is not a16× latency claim.
- Each full K256 F32 reconstructed-sum allocation grows32KiB→512KiB; two such sums are produced by the current four-term source. Their writes/reads, scalar VRF broadcast and active VRF footprints must be included. Allocator peak depends on actual lifetimes, so two allocation sizes are not a peak-memory estimate.
- Existing ratio256 max becomes32 independent ratio16 maxima, retaining one global-K scale per group. Ratio changes do not imply a measured speedup. Same final TRF payload, two weight stages, term/K reductions, F32 gather, common-scale then channel-scale then first BF16 remain.
- Constant coefficients still require physical16B packing and32B VRF collection. Whether the source generates them per group or after canonical scale extraction must be explicit; no constant-setup saving is assumed.

## Essential emitted-graph and correctness gates

1. Enumerate HBM packets and destination slices: every (c,g,q,u) reads the required original x element with no source pitch/padding overread; total logical input payload262144B. Preserve original weight bounds and native stage rows.
2. Max reduction retains Hgroup and removes only q, with ratio16 and q=0 initialized outputs. Every quantizer slice receives its own group's global4096 scale.
3. Packed terms feed the native activation TRF without a512KiB inter-group term DMA. Local term offsets0,256,512,768 and laneTerm4/K256 mapping remain exact.
4. Inspect full scale reshape and subsequent DMA: source scalar initialized, all canonical destinations covered, no padding promoted to valid data, and physical dimensions/masks unchanged.
5. TRF/VRF lifetime and enlarged live DM allocations must be checked anew, particularly overlap with first/second weight ingress. Static absence of overlapping allocated ranges does not prove runtime bank/liveness behavior.
6. Rerun the actual four-term nine-case oracle and whole-O cancellation case plus contemporary whole controls. Their current v2 PASS does not automatically validate the changed max and replica mapping.

Source contract appears expressible through existing public APIs without cluster_tile or SDK changes. Exact lowering and latency remain unverified until the selected v3 build/profile evidence exists.
