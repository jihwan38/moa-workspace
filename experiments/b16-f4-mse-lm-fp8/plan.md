# q019 MSE FFN + q018 B16 FP8 LM, default OFF

Base13174d8b is q019 with immutable MseDyadic F4 startup and physical Rc32.
The complete ops_b16_lm_fp8 kernel is copied byte-for-byte from tested q018
c79e7f3067ac161049f31f0f263576bae63a7751. Original BF16 embedding remains owned by
Weights for input embedding, prompt/scalar and R32/R64 heads. A separately owned
immutable E4 embedding/scales shadow is prepared explicitly once, with the same
q018 returned-allocation device conversion. No parallel CPU startup changes.

B16_LM_FP8_ENABLED stays false. Two per-call diagnostics select FP8 LM or BF16 LM,
then normalize dispatch to Automatic BEFORE q019 computes requested_f4. Thus both
arms keep identical q019 MSE FFN eligibility, attention, O, widths, sampling and
cache policies. Existing legacy FFN fallback intentionally remains E4 in BOTH
arms when the epilogue is disabled. Only terminal canonical B16 LM uses the new
shadow on an explicit candidate call. Existing native_q_input/native_qkv_scale
scratch is reused after final RMS; both LM launches finish before extraction.
Original F4 modules/weights, Worker, admission, and pool geometry are unchanged.

compare_b16_f4_mse_lm_fp8 derives its physical32 ownership/replay structure from
q019's existing F4 model oracle and its LM-only selector from q018. Mode model
PREFIX (512 or1024) executes three teacher-forced passes: q019+BF16 LM, q019+FP8
LM, and physical-remap replay of the candidate. Two16-owner allocations cover
32 unique rings via(13*owner+7)%32 and128 disjoint full pages. An active P128
checkpoint occupies owner15 while only owners0..14 can decode; all48 quanta
finish before width16 calls. Widths1/2/8/9/14/16, position0 reset, epilogue
fallback/reentry, all-vocabulary output/own device selection and seeded stochastic
replay are retained. Automatic at step1 must still match the BF16-LM control.
Full reserved K/V is checked after candidate against control BEFORE replay
rewrites control storage, then again after replay. Every pass refills history
from position0; no selected token changes the next teacher-forced input.

Mode guards PREFIX recomputes matched17-owner histories for explicit BF16/FP8 LM
selectors and requires scalar1 and R32-17 all-vocabulary/selection equality.
Prompt P128 logits/cache remain exact in model mode. Reports call the reference
q019 BF16 LM, not the original native-E4-FFN model. B16 approximation is reported;
replay/cache/unchanged routes are strict, and task quality is not inferred.

The probe verifies actual physical32 owning buffer extents, immutable MSE
quantizer identity, missing LM-shadow rejection, active-prompt prepare rejection,
idempotent LM allocation and the <=1.1GB added LM-shadow bound. It loads only one
12B model. Root alone runs hardware; build uses owned tail256 target with
record_run.py --fresh-package --capture-artifact. No default LM activation before
joint model gates, and no q021 public-quality claim before root's fresh six suites.
