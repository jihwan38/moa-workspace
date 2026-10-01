# E151 independent final evidence review

`independent-final-review.py --check` rechecks the frozen source/image and graph gates, three actual whole-performance allocations, extra6, the unchanged original raw-down oracle, and the original full-FFN regression. `valid` and `adoption_ready` are true; root owns live adoption.

The measured46 compilation inputs change only `src/device/shared/mlp.rs` from E150. QKV/O images remain byte-identical. E151 changes scale HBM ownership and the five Sub cast/InterTranspose4 VRF suppliers; original two native stages, F32partial/join, weighted numerical order and downstream consumers are preserved. Actual480B×7680 source packets cover3686400B once, routing30720 canonical row packets into five5760B VRFs with checked8KiB lifetimes. Added14.0625MiB F32 routing is included in whole measurements.

Three allocation paired deltas are -2.634531738%, -0.665627955%, -0.716484859%; paired median is -0.716484859%. The first allocation's elevated after-control remains in the evidence; the result is not a significance claim or an additive ranking prediction.

Correctness: nine distinct standard inputs (three repeated performance inputs plus extra6); raw-down job141395 has9 cases/34560 exact BF16 outputs with signed-zero equivalence; full production FFN job141409 has3840 exact outputs from the original13-input construction. The raw diagnostic preserves the entire production MLP as a prefix plus the original append, and is a separate diagnostic image/crate. These finite tests do not prove arbitrary-input or end-to-end model equivalence.
