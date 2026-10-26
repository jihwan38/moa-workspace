# Ownership and scope

Owned by `/root/batch_kernels`, base `a4152dba16b6b3de27b3cc474f1a890cfdb5d0a9`.
Implement only disconnected per960 activation rational quantization and packing
for P512 H3840/L15360. Preserve the original whole-row implementations and
failed complete-FFN evidence. New group scales are explicit; no quality claim.
Original dependencies, exclusive `/round2/tail256-model/target`, fresh package
and new empty `FURIOSA_OPT_OUT_DIR` per resource build. Source, host check and
exact resource compilation only. No standalone ELF, Arena or production changes.
