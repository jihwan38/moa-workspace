# E154 boundary results

Both unchanged boundary suites passed on the frozen E154 source. Raw job141647 checked8 cases/245760BF16 up/gate outputs exactly (signed zeros equivalent, nonfinite0). FullFFN job141660 checked3840 outputs exactly with maxabs0 on the original constructed regression input. Both actual upload archives, original inputs/references, source bindings and image hashes were checked; each result checker reproduced PASS.

Raw uses a separate diagnostic image with the entire42302B production MLP prefix. Its original wrapper calls the actual sharded quantizer, including Max8/directSubVRF/final512scale restore. It begins after preRMS; no intermediate RMS value was observed on NPU. FullFFN uses the actual standard whole image,299264B SHAcec8d66ee22294217dce35317024eafb728a1f09b2a089c2ce90b5f97c4e669e. The original Torch reference,13input hashes and tolerance are unchanged.

These finite targeted tests and emitted/source checks do not prove arbitrary BF16 or end-to-end model correctness. Diagnostic timings are excluded from performance. See `boundary-results.json`, `validation/probe-results-v1.json`, and `full-ffn-stress/verification.json`. No additional compile/NPU is active; m03 target and Arena slot2 are released.
