# MSE F4 B16 public-quality candidate

This distinct research branch starts at `2d0bb5a8`, preserving the default-OFF model branch and its frozen PowerOfTwo/MSE diagnostics. Original Cargo dependencies are retained, including the cleanup of local-only ring digest probes. The mathematical candidate is the same per960 F4-weight/group-scale B16 kernel already tested against intended-reference bytes and full model replay. MSE still changes original top1 in 29/60 eligible teacher-forced rows; public task accuracy is unknown.

The OFF scaffold will make enabled startup explicitly call `prepare_b16_f4_with_quantizer(MseDyadic)` and check actual physical32 owning-buffer sizes before model construction. It will not change the legacy explicit `prepare_b16_f4()` PowerOfTwo helper. A separate activation commit will enable the existing source-only B16 flag after the fresh identity diagnostic runs. Both OFF and ON will preserve the explicit original diagnostic override.

The enabled dispatch remains limited to pure independent canonical batches of 2..16 single tokens with staged/composed FFN. Scalar, prompt, mixed prompt/decode and R32 paths keep their current weights and arithmetic. No q018 FP8 LM path, request-content selector, scheduler change, quality calibration or answer-specific behavior is included. Startup publishes the immutable 48-layer shadow only after all projections succeed; original E4 weights remain resident.

Validation: scoped source/activation audit and server check, followed by a fresh-package plain server build with automatic artifact capture and source hashes. The server must log the physical32 buffer identity and MseDyadic preparation of all 144 projections before serving. The parent owns six public quality gates; performance/private evaluation waits for all six results. This branch is not root production integration and does not claim first-place or task-quality gains.

The existing OFF `compare_b16_f4` probe expects an absent shadow and is retained only as a historical/explicit diagnostic artifact; its initial assertions are not an ON server validation path. The fresh identity model/guard artifact remains in the OFF tree.
