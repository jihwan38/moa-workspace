# O row2 interleave + packed quantizer + parallel epilogue

E34 combines the row mapping proved in [the standalone row2 candidate](../output-fp8-row-interleave-two/README.md) with frozen E31. QKV and FFN source/entry points are unchanged. The O partial dot assigns rows as `120*g+2*u+s`; a typed Switch restores canonical60-row groups before the existing cluster-axis rename and H480×8 gather. The high-row rename explicitly uses the **restored canonical** slice type, not the interleaved input type. Quantization and RMS helpers retain E31 arithmetic and BF16 boundaries.

Exact O compile passes at26,220 static cycles (52instructions). This is a layout experiment, not an observed speedup. Arena job70472 passes all15comparisons at Q97092/O47308/FFN389291 cycles. O is4.31% slower than E31 repeated median45355, so this candidate is not adopted. Frozen source and hashes are in evidence.json; source delta is candidate.patch.
