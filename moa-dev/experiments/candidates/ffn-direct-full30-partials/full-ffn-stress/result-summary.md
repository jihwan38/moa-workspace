# E142 complete FFN regression result

Arena111736 passes all3,840 outputs exactly against original BF16 reference0.404296875. The original13input values/hashes and actual E142 whole FFN device image are verified in `verification.json`; reproduce with `verify_results.py --check`. This traverses repaired input preparation, the direct30 native/F32 partial path, GeGLU, weighted down and final scale/RMS/residual.

The construction is within original component value ranges, not an observed PRNG seed. These diagnostic cycles are not a performance result; one finite construction does not establish arbitrary-input or model-wide correctness.
