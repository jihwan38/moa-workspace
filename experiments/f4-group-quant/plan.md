# Per960 fixed-F4 weight quantization

Base: `8df08ec30c70729d440575476555f7a81352f692` (`b16-f4-partialscale`).
This tree changes only a pure CPU helper and its export/proofs. The parent owns
the device graph; e2e_evidence owns the complete FFN micro. No serving activation.

Input is current blocked native E4 weights, not the original NVFP4 bytes. For
each output row and contiguous K960 group choose the smallest power-of-two `s`
with `6*s >= amax`, with `s=1` for all-zero groups. Encode nearest fixed E2M1
`0,.5,1,1.5,2,3,4,6`, ties to the even magnitude code, preserving signed zero.
E4 is decoded to integer units of 1/512, so scale and nearest-code decisions use
exact integer comparisons. Nonfinite E4 input is rejected before allocation.

Physical source and fixed-code output: `[rows/240, cols/960, rows%240, cols%960]`.
Packed output uses low nibble for the even element; scales use
`[rows/240, cols/960, rows%240]`. UP/GATE use rows15360/cols3840; DOWN uses
rows3840/cols15360. The retained row dequantization and global scales do not
enter this helper and must remain at their existing device boundaries.

The intended dense reference expands each fixed F4 code to E4 without folding
in group scales. Device reference and candidate multiply each local F32 dot by
the group scale before the original inter-slice sum. Folding can exceed E4's
448 maximum (for example a quantized value512), so it is not equivalent.

Gate: standalone Rust tests plus independent Python Fraction oracle for every
finite E4 code and every reachable scale, exact nibble/layout checks over
multiple row/K blocks, true row/column permutations and rejection cases. The
subsequent device gate must compare all retained stages to a dense same-code
reference, independently sample F64 partial dots, report error against current
E4 and original provided weights, and time the complete FFN. This CPU gate
cannot establish task accuracy or hardware throughput.
