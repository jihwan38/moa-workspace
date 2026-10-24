Control is four established materialized-input P128 native QKV groups from
318102, including one row quantization pergroup. All input/scale/output buffers
remain separate and each request computes every row independently.

Candidate ops_qkv_native_four accepts four existing P128 input/scale/output
pairs. It reads the original E4 weight into the same H960 DM mapping once, then
executes the unchanged TRF64 prequantized core four times with that retained
weight. No host row pack/split, physically blocked activation experiment or
new quantizer is included. Dense agent owns the separate blocked P512 IO work.

Check all four output groups bit-exact against current P128 prequantized cores
and independent bounded quantized-input F64 oracles, including permuted group
order and distinct row scales. Time complete four-group slidingQKV and fullQK
sequences, including all four original quantizer launches. Keep host group
factoring inactive unless this removes enough time to matter. No production
precision or dispatch policy changes are part of this isolated test.
