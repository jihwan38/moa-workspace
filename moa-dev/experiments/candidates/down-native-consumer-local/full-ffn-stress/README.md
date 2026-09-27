# E140 same-input whole FFN precision regression

Preparation only. Reuse the unchanged E134 regression host and official-reference
BF16 output (SHA256d2930beb2df2274c5b316865d4563e4babe550c1ef930cf2bbcd02cca79f3a74).
Build against frozen E140 whole sources; require its embedded FFN image to equal
the ordinary exact whole build. The unchanged host creates the same13 original
input tensors, invokes public ops::decoder_feedforward, and requires finite
outputs under original0.01+0.01*abs(reference) tolerance. This runner additionally
requires all3840 exact outputs on this selected case. Timing is excluded.

Original reference generation, PyTorch implementation and independent input
codegen checks remain in ../ffn-native-down-weighted-residual/full-ffn-stress.
This constructed input is within the generator component ranges but is not a
claimed generated seed. This is a finite regression, not all-model correctness.
