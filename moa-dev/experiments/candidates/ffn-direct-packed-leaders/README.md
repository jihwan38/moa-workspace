# E155 completed: no adoption; E154 remains the runtime

Three independent E154-bracketed whole allocations passed all135 comparisons but did not establish a net improvement. The frozen first positive screen is preserved.

| Job | E154 before / after | E155 FFN | Paired change |
|---|---:|---:|---:|
|141722|288282 /289016|288517|-0.045730%|
|141733|289264 /287884|288455|-0.041237%|
|141747|287806 /288598|288339|+0.047536%|

The paired median is−0.041237%, with mixed signs around zero. None of the three candidates beats both individual controls. The root rejected this prototype because the improvement was not established, not because it was below5%.

The actual uploaded stage and outer batch bytes are identical across allocations. Source46, one MLP change/45unchanged, current E154 control and QKV/O image identity are verified. Extra6 job141755 passes30 comparisons, bringing the standard-input scope to nine distinct inputs. Raw8 job141734 passes245760 exact first-BF16 values (signed-zero equivalence); full-FFN job141746 passes3840 exact values. Raw exercises the quantizer/routing but excludes production pre-RMS. This finite scope is not arbitrary BF16 or whole-model equivalence.

The removed late packed destination is23040B. Seed184320B and ring5898240B remain; CL1 logical crosscluster destination rises11520→92160B. The actual descriptor covers all16leaders from initialized CL0 packed shards. These bytes are costs, not cycle savings. The current composition completes; it does not identify the cause of old M06 timeouts.

Decisions: no adoption, stop parameter tuning and extra measurements, retain the complete-path evidence for later research. The verified E154 source/runtime and official submission remain unchanged. Provenance files named `measured-build-e155-adoption.json` describe the measured source/package and explicitly do not adopt it.

[Final independent review](adoption/final-no-adoption-review.md) · [Measurement decision](decision.json) · [Performance](adoption/performance-summary.md) · [Boundary scope](boundary-result-summary.md).

Reproduce:

```sh
python3 moa-dev/experiments/candidates/ffn-direct-packed-leaders/validation_performance.py --check
python3 moa-dev/experiments/candidates/ffn-direct-packed-leaders/validation_extra.py --check
python3 moa-dev/experiments/candidates/ffn-direct-packed-leaders/provenance.py --check
python3 moa-dev/experiments/candidates/ffn-direct-packed-leaders/adoption/final-no-adoption-review.py --check
```

The following is the historical preparation record.

# E155: CL0 packed terms feed sixteen ring leaders directly

Actual start f33e7c6d723cfc96a360e23a6053ce74481a5a34, source/control adoptedE154/3f384063f5231e00ad286f365209806e5043fc83. OnlyMLP changes,45other
compileinputs and QKV/O images must remain identical. Remove the23040B intermediate
CL2H480 packed copy and let the original typed leader DMA read CL0's initialized
8H480×3term shards. Keep184320B seed and5898240B ring, all arithmetic/scale8/directVRF
and E154native/down. CL1crosscluster logical payload increases11520→92160B(+80640).
Actual descriptors, source1440B bounds, all16leader coverage and registerlifetimes
are mandatory before the wholeE154bracket. M06v4/v5 compiled this routing but timed
out; their cause remains unknown. This new surrounding composition requires its
own NPU/correctness evidence. It is a local routing replacement, not a new family.

One bounded deadline attempt: negative/timeout stops; positive freezes source and
completes3alloc,extra6,rawup8/fullFFN. E154 remains the verified official fallback.
[Recipe](direct-v1/recipe.json) · [Prepared source review](selection-review.md).
