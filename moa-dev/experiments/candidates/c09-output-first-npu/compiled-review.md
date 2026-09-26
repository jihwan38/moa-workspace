# C09 first-NPU compile: pipeline and gather handoff review

2026-09-27. Read-only review of `c09-output-first-npu/evidence/c09-first-npu-v1/{graph,schedule}.json`. No compile/Arena operation by this reviewer.

The new exact image is byte-identical to frozen C09 two-stage-v1:80,640B, SHA256 `fe90ed758c27fe40a11e78ff5132e7b98152fc8cd79208d49c22053672859495`. Frozen projection SHA256 `5692369dd5b8f6ddbeecdfaa9289a9683fae18798b920a71775362f306560fe6` is preserved. This is the first hardware measurement of that prototype, not a repaired or tuned pipeline.

The compiled structure contains both60-row weight stages, both Main lookup/dot/16-column F32 reduce stages, one shared activation TRF, two table setup sequences and one canonical gather. The native image's26 engine marker kinds match all26 scheduled engine operations in order. The original source and same-image marker plan are `/tmp/moa-c09-marker-plan.md`.

## Handoff finding: distinguish a new dependency chain from a pre-existing sync pattern

The baseline already has the canonical gather destination path:

```
baseline Scratchpad#O9 → ClusterSync#O10 → gather DmaCommand#O11
baseline dot → LowLevelReshape#O8 ──────────┘
gather → ClusterSync#O12 → channel scale
```

C09 contains the same kind of destination preparation:

```
C09 Scratchpad#O26 → ClusterSync#O27 → gather DmaCommand#O28
C09 IndexWrite#O24 → LowLevelReshape#O25 ─────┘
gather → ClusterSync#O29 → channel scale
```

Thus **the presence of Scratchpad→ClusterSync before gather is not newly introduced by C09**. The added mutable-result path is separate:

```
result Scratchpad#O10
  → IndexAccess#O12 → first dot O16 → IndexWrite#O17 (T22)
  → IndexAccess#O19 (T22, row60) → second dot O23
  → IndexWrite#O24 (T30) → reshape O25 (T32)
```

The final gather O28 takes data T32 and destination-sync T35 as separate inputs. There is no direct graph edge from IndexWrite O24 into destination ClusterSync O27. It would therefore overstate the evidence to say “IndexWrite inserts a new ClusterSync” or to attribute a precise sync cost solely to IndexWrite. The graph establishes additional mutation/view dependencies; the schedule establishes a later gather handoff.

The baseline destination ExplicitSync is scheduled1203→1803; C09's corresponding sync wrapper spans1206→20413. Wrapper lifetime includes waits and is not its isolated service cost. C09 final dot ends20207, IndexWrite bookkeeping ends20213, and gather begins21413. The actual measurable boundary is **final dot end→gather begin=1206 compiler cycles**, versus0 in baseline. Trace should preserve all generic wrappers and measure this boundary without assigning the entire interval to a single SDK operation.

## Static pipeline arithmetic

| Boundary / interval | Baseline O | Frozen C09 |
|---|---:|---:|
| First weight begin |2774|2774|
| First dot begin |16157|9740|
| Final dot end |21220|20207|
| Gather begin |21220|21413|
| Gather interval |470|470|
| Whole makespan |27239|27432|

The C09 final dot is1013 cycles earlier, but the1206-cycle handoff leaves gather193 cycles later. The whole makespan is therefore+193 (+0.70854%) even though useful pipeline overlap exists.

At the ingress side, table1 DMA38 runs9740→10578 while dot0 runs9740→12403. Weight1 DMA42 starts10578, ends17544 and overlaps dot0 by1825cycles. Table1 preparation ends11871, before weight1 completion. On the static schedule it is weight1, not table1 preparation, that directly precedes the second dot. Whether table1's DMA placement or actual resource waits reduce useful overlap needs the NPU trace.

This motivates three separate hardware questions:

1. Does W1 actually overlap dot0, and how large is the W0-end→W1-begin interval containing table1 DMA?
2. Are smaller weight/dot tasks and repeated table preparation more expensive on hardware than the compiler predicts?
3. Does final mutable-output completion leave a significant last-dot→gather handoff, even if the last dot finishes earlier?

The analyzer now reports those stage-specific windows, common first-weight→final-dot/gather-end boundaries, canonical gather and final tail in the same cluster Task clock. It separately records `*_sum_diagnostic_only` resource-duration sums, which are never interpreted as latency/savings. The control120-row dot and C09's first60-row dot are not directly compared as equal work.

No NPU outcome is assumed here. A regression of this frozen combination should not be expanded to a rejection of all supply/compute pipelines. Conversely, observed overlap is insufficient for adoption without complete-O correctness and gain against contemporary controls.
