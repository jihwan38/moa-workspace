# M05 — M03 ownership with producer-local activation preparation

Branch `opt/m05-local-activation-ready`, parent `db7cbf4` integrated into main.
Crate source origin remains E74 `9e2058ec0947498d7cb693fbe98e7b5f0b5c9ea7`;
`local-prepare-v1/recipe.json` pins the actual M03 and M04 component snapshots at
`db7cbf4cab1cb2a5e72b937d649fc4a2fbac0283`. E104 is the independent performance
control. M05 is a **M03/M04 recombination with local activation dataflow replacement**,
not a new independent architecture family or a new tile-size experiment.

## Source path and specific untested combination

Unchanged `ops::decoder_feedforward` calls pre-RMS, then `feedforward_sharded`.
The changed entry no longer copies normalized BF16 x to two clusters before amax.
It calls M04's `prepare_ffn_fp8_split` on the original cluster0 input. Amax, scale,
and three E5M2 terms are computed once. `broadcast_ffn_quantized` copies these
prepared terms to the ring leaders in both clusters, then Main Switch replicates
them to512 slices. Native up/gate consume the same representation and scale.

All subsequent M03 code is retained: three complete weight-materialization and
dot streams, cluster-local 120-row GeGLU packets, four-column L-owned down,
and two F32 H3840 cluster partials summed before BF16/global scale/epilogue.
The M04 five-row-tile streams per projection are excluded. Public shapes,
host-generated inputs, fixtures, tolerances and the final epilogue are unchanged.

```text
pre-RMS, cluster0 -> local amax/scale/three-term E5M2 input preparation
 -> prepared input/scale distribution to both clusters -> ring32 replicas -> TRF
 -> materialized native up/gate: each cluster owns its L7680 half
 -> local F32 compact -> BF16/scaled GeGLU -> local down input distribution
 -> BF16 down/block scale/four-column F32 sum -> gather two H3840 partials
 -> cluster sum -> BF16/global scale -> post-RMS/residual/layer scale -> HBM
```

This is the first ledgered combination of M03 ownership with M04 input preparation
but full-projection weight streams. M03 measured1.36% faster while retaining the
initial activation-copy dependency. M04 measured13.67% slower with both changed
input preparation and fifteen row-tiled decode/dot streams. Neither experiment
isolated this combination. Their percentages cannot be added to predict M05.

## Ownership, lifetime and costs

- Normalized BF16 H3840, amax and restoration scale initially stay in cluster0 DM.
  The same arithmetic as M04 prepares three E5M2 H480 pieces on eight quantizer
  slices. F32 scale/term inputs reside in VRF only while the corresponding Sub/Main
  producer consumes them. No input conversion is performed free on the host.
- Completed terms occupy3×3840 bytes before replication. The typed DMA sends full
  terms to eight ring leaders per cluster; Switch then fills all512 slices.
  Final term replicas and TRF size remain unchanged. Cross-cluster transfer is
  deferred, not eliminated. The allocator/scheduler may move the wait later.
- M03 up/gate owns `l=7680*c+30*s+t`. FP4 DM57,600B/slice expands to E5M2
  DM115,200B, native DPE writes F32 block partials, and VRF6×240 scales reduce rows.
  Compact120-row GeGLU remains within the producing cluster.
- Down owns `s=4*r+k`, `h=60*r+t`, `l=7680*c+1920*k+u`. Local weight60×1920,
  FP8 intermediate, F32 partials and VRF12×120 scales are unchanged from M03.
  Final two H partials occupy30,720 logical bytes before their cluster reduction.

Removed: the raw input cluster-copy dependency before amax and duplicate
quantization. Excluded: M04 repeated tile DMA/lookup/dot setup. Retained: all
three full weight loads/materializations, all final activation replicas and M03
output-partial gather. No claim of a42k-cycle saving: prior trace intervals include
queue/dependency waits and overlap with other work.

The specific mechanism is to make activation computation independent of the
initial small cross-cluster DMA, which previously followed the first large weight
load. Rust source order is not a scheduling guarantee. Check the emitted order
and complete NPU runtime; a smaller static gap alone is insufficient.

## Gate and evidence plan

Build the full FFN and unchanged benchmark host. Arena order is E104 before,
M05, unchanged M03 companion, E104 after. M03's exact existing package is reused
with its original build record/hash; it is not recompiled or retuned. Candidate
comparison remains E104; the companion measures the combination's incremental
effect under the same allocation.

Close a regression or target gain below5% without local tuning. Only a promising
complete candidate gets independent repeats/expanded correctness and adoption.
Basic validation of either component is not inherited by the combination. Keep
all source/package/log evidence, including failures. Official submission requires
an adopted validated candidate; this screen is an Arena experiment.

## Result — E118 closed below the screening cutoff

First source compile and host build pass; exact/host FFN images match. QKV/O
images match E104/M03. Arena104458 completes all60 unchanged basic comparisons.
Candidate FFN3 inputs/11520 outputs pass. One allocation, no expanded accuracy.

| Process | FFN median cycles |
|---|---:|
| E104 before | 335775 |
| M05 | 335317 |
| Original M03 companion | 331749 |
| E104 after | 337758 |

M05 observes−0.43% versus the bracketing E104 median and **+1.08% versus M03**.
Close without tuning, adoption or official submission. E104 stays live.

Emitted pre-RMS→amax gap does disappear (23250→0), but complete term broadcast
ends56931→60233 compared with M03. Entire schedule154440→156992; DMA37→36.
These are compiler intervals, not measured NPU internal timings. Moving one
initial dependency did not make the complete operand path faster in the screen.

The next concrete hypothesis is to consume pre-RMS's existing H480 shards
without canonical gather/repartition, and remove scalar/term packing copies
where supported. This is a separate unimplemented combination at M05 closure,
not a claim of a working10th-place kernel. `decision.json`, `verification.json`
and immutable sources/logs retain the negative result.
