# Exact dense full-attention state integration

Base30038f5 includes current norm fusions and validated direct full attention.
The passed a9eefd4 kernel (Arena318736) changes only internal max/sum transport:
query-pair transpose, head coalescing, dense HBM Stats[P128/64,Gf,P128%64], and inverse
load. Original padded Scalar arithmetic and all BF16/page recurrence boundaries
remain. Page-byte-offset metadata is reused unchanged; Workspace/KV storage has no
change. WidePrefill adds two8192-byte scratch tensors. Old padded tensors remain
for legacy/partial paths and both earlier diagnostic controls.

Serving default is an ordinary Rust constant. PreparedWide owns the selection;
all first/next pages and final divide choose the SAME dense mapping only when
canonical+direct+compact conditions hold. Current chunk commit precedes full
attention, chronological Blocks.full mapping and final causal mask stay intact.
Group/Prompt512 inherit each prepared state's setting and keep their ordering.
No API handlers or Worker dispatch changed.

New forward_sequences_direct_full_attention disables only dense transport in a
fresh local state, retaining the passed direct kernel, native QKV and all other
current math. Existing original-full diagnostic still selects the pre-direct
kernel. compare_full_stats_attention2049/17 checks every128prefix, scalar tail and
four teacher-forced rows with strict equality; unchanged compare_prompt512896/1
checks48independent R64 interleaves, final prompt and four postdecode rows.
All smaller tails follow the same ordinary path. Partial-wide diagnostic QKV
mode differences are not used as a control.
