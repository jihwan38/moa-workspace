# E151 actual whole scale-routing gate

`audit.py e151-ingress-v1 --check` passes against the frozen exact source, graph, schedule and image. Only the original E150 scale DM ownership and five Sub VRF-producer source regions differ; all other45 compilation inputs and all numerical operations are preserved.

Actual scale DMA emits7680 terminal packets of480B, covering the original3686400-byte E4 scale matrix exactly once. Source slice4g+b owns row60g+4r+b with compact480-byte local row. Each three-row source view reads the actual offsets `480a+120q+u`, then Sub InterTranspose4/time3 emits row12i+4a+b to consumer slice4g+q. All30720 destination row packets are bijective and preserve the120-element column order.

The five resulting F32 VRFs are5760B/slice, each at[1472,7232) with nonoverlapping reuse lifetimes. The whole schedule's VRF allocation and every VRF/TRF consumer lifetime fit8KiB; no extra bitmap operand is present in these five InterTranspose streams. Original native-half weight descriptors and weighted arithmetic, disjoint F32 partial writes, both-write join, and all five12-row scale/reduce consumers match E150. Rows24–35 still cross the two native halves correctly. Fixed up/gate480B copies and ring32 bitmap bounds also pass.

This proves the compiled routing/extent and lifetime contracts. It does not predict performance, exclusive Switch/NoC cost, or NPU numerical correctness. Existing raw-down and whole-FFN suites plus contemporary whole measurement remain separate gates.
