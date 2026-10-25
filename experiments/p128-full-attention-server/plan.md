# Exact full-page movement integration

Base6d58608 includes the current validated head-major sliding attention, FFN input
reuse and B16 output changes. Kernel source is the unchanged9c7b75a full-page
candidate, whose repaired1603567 probe passed Arena318664. The default is an
ordinary Rust constant in prefill_wide.rs, so a plain src-only serving build uses
this path without Cargo features or evaluation-specific configuration.

Only native complete128-row canonical full attention changes. It still commits
current K/V before reading full pages, visits logical pages in ascending order,
resolves every physical page through the execution-owned Blocks clone, applies
the causal additive mask to only the last page, and uses the original final divide.
No cache migration, page fusion, model precision or request ownership changes.
Partial B16 segments and legacy Vec pages keep their previous implementations.
Group and Prompt512 inherit the same page-local128-row continuation in their
existing chronological order, including deferred output projection.

Workspace adds immutable per-physical-page HBM index tensors[P128/64,Gf]; all32
entries equal physical_page*Tf*Df*2. The canonical256-page pool stores32768 bytes
of index payload, allocated once during initialization. This avoids per-chunk
index allocations/uploads. Legacy workspaces keep the vector empty and cannot
enter the canonical branch. Layer KV tensors remain separate owners.

PreparedWide owns the direct_full_attention selector. The explicit
Model::forward_sequences_original_full_attention diagnostic changes only this
selector in a newly prepared local execution; no global state is changed or
restored. Native QKV and the validated sliding path remain identical in control
and candidate. Existing active-allocation guards and error restoration stay on
the shared Model path.

Gates: existing host tests; plain server/device build; compare_full_direct_attention
2049 0 checks every128-token prefix, scalar tail, and four teacher-forced decode
rows bit-for-bit with reversed disjoint physical pages; unchanged compare_prompt512
896 1 checks chronological ring/page boundaries and independent R64 interleaving.
No numerical tolerance was added to either whole-model gate.
