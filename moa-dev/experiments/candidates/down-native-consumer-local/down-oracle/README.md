# E140 actual down boundary validation

Prepared for the consumer-local weighted-down candidate; not yet compiled or run.
The immutable E134 host, wrapper and nine mathematical-reference cases are reused.
The E134 log labels identify that unchanged host; the build and actual production
source/image binding identify the E140 implementation. No old result is inherited.
All34560 outputs must match exact BF16 (signed zeros equivalent), including the
known tiny-value regressions and independent scales for the two L halves.
The new canonical BF16→K1920 ownership and K4max must be verified on hardware.

The production MLP is copied byte-for-byte, with only a diagnostic entry appended.
The wrapper invokes actual project_down_tile and finish_down from original BF16
L15360, FP4 weights and original E4 block scales. It does not run the FFN prefix
or epilogue; the separate unchanged whole-FFN image stress test covers the known
full-output regression. These finite tests do not prove arbitrary BF16 accuracy.
The recipe is written only after the exact whole build and graph gate pass.
No diagnostic timing is performance evidence. Record files are immutable.
