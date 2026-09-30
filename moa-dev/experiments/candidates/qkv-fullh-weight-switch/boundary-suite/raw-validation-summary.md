# E144 raw9 production-boundary result

Arena **136561 passed all9 cases,27 Q/K/V comparisons and73,728 BF16 outputs exactly** (±zero equivalent), with zero mismatches and zero nonfinite values. `raw_oracle.py --check` passed after collection. All63 actual Rust-generated input SHA256 values match the original CPU record; wrapper, host, F64 generator and regenerated expected-truth bytes are unchanged from the successful original9-case suite.

The diagnostic contains the complete frozen `fullh-v2/projection.rs` (SHA256 `3d2500295785592bebdc723fb62617398e2e001ba41c63b2a5c078cda5e00fed`). Its46 original compile inputs are preserved, except the documented `lib.rs` module append; two diagnostic source files bring the ledger to48. The selected whole QKV image is206,592B SHA256 `687f1c8df3ccf24eee1e9425a64c685148620e09c4a245d4b039231380f3bc53`. The separate raw diagnostic image is134,656B SHA256 `39ac16920a3a26e20e6a2a6095410ed8dff161395127a7e1a8a83bd8d1eeb261`; its exact-compile and host-build bytes match and occur once in the actual uploaded host.

`raw-results-v1.json` binds the actual uploaded archive and original raw log. All payload files match the frozen package. The downloaded job manifest also names `beef-e144-qkv-raw-v1`, `./run.sh`, no arguments/environment overrides and70-second timeout. No retry or source repair was performed during this validation.

The requested scale-dpe target did not exist. Root assigned the existing `/tmp/moa-c01-upgate/target` exclusively; it was released after the successful host build. Arena slot2 was released upon successful completion, before root's separate full2 submission.

This validates the selected post-preRMS BF16 -> unchanged production projection -> pre-head-RMS BF16 boundary. It does not execute the complete public preRMS/head RMS/RoPE/cache path, establish arbitrary BF16 accuracy, or measure performance. Those claims require the separate whole2 and standard whole-kernel records. See `raw-validation-summary.json` for input digests and exact counts; the original preparation record remains historical and unchanged.
