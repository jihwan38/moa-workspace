# q022 fixed-E4 sliding KV quality-only surrogate

This separate research artifact starts from verified ON375bdcf4, clean physical32 c0813d497087. Runtime source changes from the proven ON probes are comments only. Fixed scale1 BF16→F32→E4→BF16 is applied at all ten sliding K/V producer implementations before current-chunk attention or cache commit. Q, full-attention kernels, model weights, original FFN/LM, BF16 pool/reader ABI, real-owner masks, scheduler and sampling are unchanged. There is no q018/q019/q020/q021 model composition or compressed-storage port.

All ten ON producer byte gates passed; fresh OFF negative controls passed after fixing the invalid shared device-cache control. Same-program model512 all-logit/cache replay passes, but corrected original-vsrounded comparison changes34/72 top1 IDs and has mean0.1867 relative L2 over128 fixed samples only. This is not an all-element error norm or public task quality. Remaining scalar/R32/tiled/packed structural gates continue under the root agent.

Building before those gates finish is preparation only. Root may run six public quality suites only after structural acceptance; no performance or adoption claim is authorized by this build. The surrogate intentionally retains BF16 storage, so its timing is not compressed-cache throughput.

Build uses the original Cargo.toml/lock, fresh-package source capture, a new empty documented FURIOSA_OPT_OUT_DIR, and immediate target/release/server capture. Kernel registry must match all common entries in the verified ON model/packed probes. This independent output directory avoids the observed SDK0.9 constant-only shared-kernel fingerprint collision; preserved a005/a006/a007 artifacts and detailed incident are in fixed-e4-kv-quality-model evidence. The original OFF source remains available there.

Server startup retains physical32 public owning-buffer identity validation before Model::new. No runtime precision toggle exists, so every request history has one precision from token0 and no per-request content routing is introduced. Root alone freezes q022 and dispatches Arena/public tasks.

Here unchanged Q/full means the local producer operations and full-attention implementations are untouched. Later hidden states and full-cache contents can change indirectly after earlier sliding layers; no original full-cache value-equivalence claim is made.
