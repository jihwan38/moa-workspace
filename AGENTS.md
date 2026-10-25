# Ownership and scope

Owned by `/root/e2e_evidence` for full P512 H3840/L15360 exact-rational
quantization plus packing. Base: `814c33b7`.

Keep validated standalone quantizers and original packing controls unchanged.
Only new disconnected fused helpers/wrappers, imported original pack controls,
lib exports and experiment evidence may change. Preserve the independent oracle,
original Cargo dependencies, model, and official candidate. Reuse only
`/round2/prefill-head-attn-commit/target` and a unique empty kernel output directory.
Authorization is source/check/resource only; no device ELF, Arena or model
activation. Root alone runs Arena.
