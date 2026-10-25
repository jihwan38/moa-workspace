# Exact B16 output cluster split

Source3105678 integrates the passed a611b58 `ops_output_b16_hsplit` kernel at all four narrow-prompt/B16-decode output calls. The output projection distributes the3840 output channels across two clusters via independent Ho=2,Hb=30,Ht=64 axes. Host-side mutable tensor views preserve the existing B×H allocation. The K1024 BF16 dot partials, pairwise BF16 addition tree and channel scale are unchanged. `Model::set_b16_output_hsplit` selects the retained old kernel for diagnostics; serving defaults to the split kernel.

The strengthened independent NaN-buffer and F64 microprobe318485 passed. The same-ELF whole-model gate318509 compared48,758,784 BF16 logit values exactly, with64 actual512-token histories, remapped physical slots/pages,64/17/16/4 active batches, page transitions, position-zero reuse, and a15-token prompt interleave. Both passes keep every other kernel and token fixed. Individual model timings are diagnostic, not a controlled HTTP performance claim.

Frozen candidate `c035-b16-output-split`, built by `r487-b16-output-server-build`, SHA256 `1ca063c4b7d69398e1e426e9ef752d7ae1e7011505987f69e34a0f94a3fec273`:

- Public mixed25/C5 job318511:45.976576 generated tokens/s; TTFTp90 3674.206ms; TPOTp90 87.0663ms. The prior c033 matched configuration measured43.78711 tokens/s,3936.856ms,89.9629ms. Separate runs, not an official score.
- Public GSM34/C17 job318512:34/34 passed in68.018s including startup. This is close to the70s Arena limit; use a frozen one-round harness for future17-concurrency regression probes when a second round adds no needed coverage.

The official submitted checkpoint remains c031/25b712dd40.2445 tokens/s. This candidate has not consumed an official submission. Final rank/top3 remains unverified.
