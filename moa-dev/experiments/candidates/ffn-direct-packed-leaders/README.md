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
