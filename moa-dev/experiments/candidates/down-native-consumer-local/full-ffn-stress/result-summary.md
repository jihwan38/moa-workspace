# E140 copy-only complete FFN stress

Arena111488 executes the exact `control-copy-fix` public FFN device image. All3,840 outputs equal original BF16 reference0.404296875. The13 original inputs, host, reference, compilation and uploaded archive are bound in `verification.json`; reproduce using `verify_results.py --check`.

This is the unchanged E134 construction within public component ranges, not an observed PRNG seed. It traverses the repaired upstream quantizer, both projections, GeGLU, weighted down and final scale/RMS/residual. It is a finite correctness check; diagnostic cycles are excluded from performance claims.
