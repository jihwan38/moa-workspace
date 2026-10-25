# Exact P512 single-launch FFN host integration

Base: root `4f0be45` (c056 normalization shard quantization). Device implementation
is copied from measured `a9d9311`; micro result 318883 is recorded in batch tree
`afff9ec`. The current four-launch core remains the matched diagnostic control.

`PreparedGroup::fused_ffn_core` defaults true and belongs only to the prepared
execution. The explicit original-FFN constructors disable only this flag; all
fused normalization, H480-local norm quantization, attention, output projections,
postnorm and request ownership remain identical. A completed or cancelled
prepared state does not leave a persistent model mode.

The candidate consumes and writes the existing GroupPrefill-owned q_input,
input_scale, up, gate, q_hidden, hidden_scale and projected buffers. It adds no
scratch allocation and replaces only up, gate, GeGLU+quantization and down's
four launches. Existing layer-yield and finish behavior is unchanged.

`compare_p512_ffn_core [start=896] [interleave=1]` compares matched chronological
Prompt512 executions in disjoint rings/reversed full pages. It retains existing
ownership rejection checks, 48 independent R64 interleaves, and four subsequent
decode tokens. Both passes preserve the current norm/shard path.

`compare_p512_ffn_core_group [896 384 0 128] [interleave=0]` compares matched
four-owner grouped executions, each with the same four independent P128 chunks.
Control and candidate rebuild their causal histories into disjoint allocations;
the control uses four launches, not four serial P128 executions. Optional48
R64 interleaves and post-group B16 decode check shared scratch/cache boundaries.
All vocabulary logit values must be finite and bit-identical, including signed
zeros. Timings are diagnostic; hardware dispatch is coordinated by root.
