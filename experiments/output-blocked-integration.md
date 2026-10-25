# Exact grouped sliding output storage and direct norm consumer

Source producer89a841c, consumerad2d172. Kernel318788: all blocked and restored output bytes exact over2seeds2orders4outputs; warmoriginal.522ms,blocked.373ms,blocked+restore.519ms. Thus no restore path is adopted. Consumer318796: all512FP8rows/F32scales/residuals exact vs canonical fused norm, CPU F64 passes; canonical~255-289us,blocked~268-300us. Integrated whole-model timing, not subtraction across runs, decides adoption.

GroupPrefill owns four typed blocked output buffers. Sliding attention and KV chronology are unchanged; after all four attention/cache updates, O writes blocks, the exact H960 bridge/H480 norm consumer reads them immediately, and independent residual checkpoints persist before FFN. Full attention uses canonical O. Original-norm diagnostics use canonical O; an independent prepared-state original_output flag allows a matched full-fusion control. No global setting or shape guessing. Wider decode scratch is separate and untouched.

Required before serving evaluation: host tests, matched original-output/fused-output same Prompt512 all53logits exact, four independent owners with interleavedR64 and postdecode, then long mixed performance. No official submission from this source until coherent public accuracy gates.
