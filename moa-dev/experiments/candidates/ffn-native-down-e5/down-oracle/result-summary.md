# Actual-down oracle result — job109784

[Raw results](results.json) and [raw log](evidence/arena-109784.log) are preserved.
The job completed in2.907s with exit0. The complete source/image/upload recheck
`python3 .../down-oracle/probe.py --check` passes.

Seven exact-representable mapping/algebra cases produced26880 BF16 outputs,
all exactly matching the original-BF16 reference (signed zero equivalent).
They include different cluster scales, input/row boundaries, zero and block/
cluster cancellation. This validates those targeted cases through the actual
frozen v2 down helper; it is not proof of generic BF16 equivalence.

Adversarial case4 produced a quantified loss in all3840 rows: original-BF16
reference1.40625, candidate0.703125, absolute error0.703125 (50%). This exceeds
the diagnostic0.01+0.01×abs(reference) tolerance. The cluster with amax128 loses
its2^-24 values in the E5 representation, while the low-amax cluster retains its
contribution. The large amax terms themselves have zero weights, so the lost
small-value contribution is not obscured by an unrelated large dot product.

`results.valid=true` means the bound diagnostic executed, its seven required
mapping cases passed, and the source/image/archive evidence agrees. It does
**not** claim all eight numerical cases pass. The record separately contains
`adversarial_tolerance_failures=1` and `full_ffn_followup_required=true`.
The adversarial vector is synthetic and its reachability through the unchanged
complete FFN/public input distribution has not been established by this probe.
A raw down failure therefore is neither an automatic rejection under the public
end-to-end tolerance nor permission to hide the approximation. Root must resolve
its full-FFN relevance before final adoption, using the original acceptance and
later scale/residual/RMS operations. No such follow-up was run by this agent.

The diagnostic image is112640 bytes, SHA256
`75c5d91e624abdfd92c84c2865238b661dc55ce263e703a734c50c9a660c137d`.
The actual uploaded archive SHA256 is
`cf5c53705be316a3de7c8aa3e49bfcb90370c8f60f2c88ddd77b4334feeecbba`.
Production v2 source SHA256 is
`c3ae74b55b306742b69b438bf2ea603409a0644c55f9004d7fe01665c656a5d3`.
The warm build target was released immediately after exact/host compilation.
This oracle contributes no kernel-performance or official-submission result.
