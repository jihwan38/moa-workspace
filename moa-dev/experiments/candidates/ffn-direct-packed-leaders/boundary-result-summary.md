# E155 boundary results

Both original boundary suites passed on frozen E155. Raw job141734 checked8 cases/245760BF16 up/gate outputs exactly (signed zeros equivalent, nonfinite0). FullFFN job141746 checked3840 outputs exactly with maxabs0 on the original constructed regression input. Actual upload archives, unchanged inputs/references, source bindings and image hashes were checked; both checkers reproduced PASS.

Raw has a separate diagnostic image with the entire42196B production MLP prefix, and exercises the actual CL0packed→16leader typedDMA followed by the original ring/native path. It begins after preRMS, whose intermediate values were not observed. FullFFN uses the actual standard whole image,299776B SHAbfc36bee83a766974efe12c0db4656389ebae30ff26ab5f5bd01b047e1e8327a. Original Torch reference,13input hashes and tolerance are unchanged.

This is targeted correctness/provenance evidence, not a performance/adoption decision or general BF16/end-to-end model proof. Source removes23040B intermediate but increases CL1logical crosscluster destination11520→92160B; these diagnostics do not measure its speed. See `boundary-results.json`, `validation/probe-results-v1.json`, and `full-ffn-stress/verification.json`. No further boundary compile/NPU is active; m03 and slot2 are released.
