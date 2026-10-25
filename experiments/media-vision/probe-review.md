# Read-only review

Agent round2_path reviewed frozen probe2c1d442. It found no control/candidate buffer reuse or timing arithmetic blocker. Fused candidate, staged batch and scalar outputs are independent; all stage destinations are checked. Inclusive ABBA counts position gathering, uploads, output allocations and completed kernels, and excludes downloads/scatter consistently.

Limitation: ScalarBuffers::new performs all scratch allocation/uploads before the three launches. Current runtime::encode_vision_patch interleaves allocations with those launches. The counted work is equivalent, but this microbenchmark is not exact serving chronology and is not an end-to-end throughput claim. Kernel-only timing is unaffected by this ordering difference.

Every fixture row is computed and compared, including zero-padded rows for active7/1; no skipped padded work or partial-batch latency claim is made.
