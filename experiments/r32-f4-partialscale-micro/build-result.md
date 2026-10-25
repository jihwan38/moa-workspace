# R32 MSE F4 micro ready for root hardware gate

Functional6f6c0de5, base98a6d7e5/physical32 control07718f07. Host typecheck
passed13.08s; release device build passed79.87s. Every source hash matched
before/after both commands. Candidate core bytes match resource1a4a57a3;
encoder bytes match8e7ce99e; mechanical intended reference changes only
F4→E4 weight dtype and removes fixed LUT (plus descriptive header).
Original norm/core source files are untouched.

Frozen ELF: `artifacts/test_r32_f4_partialscale-6f6c0de5`,4,905,640bytes,
SHA35a203a04eb4c4a0de3e5eceecbaeeda6bde615d4d1f189ae666d7ebfb3de1a5.
Two suggested70s jobs: arguments `0 2` (layers0/5), `2 4` (layers6/47).
No NPU run has been performed; root alone dispatches.

The complete timing graph is exactly three current calls (pre-RMS, four-stage
core, post-RMS/residual/gate). Speed threshold>=10% is asserted only after all
numerical/timing output; numerical PASS is labelled separately. Current and
candidate calls use the same unchanged normals and actual retained weights.
No post-attention call is added to this FFN region.

All eight retained buffers are checked bitwise against independently uploaded
dense fixed-code weights with the SAME group scales. Original-E4 drift is
reported over first32 rows; inactive R64 tails are separately checked raw.
Independent F64 samples use the original B16 tolerances and separate decoders.
No claim of F64-exact accumulation or model quality is made.

Independent e2e_evidence source review passed: physical activation indexing,
blocked hidden permutations, live17/31/32 bijections, matched feedback, NaN
normalization tails versus untouched projected sentinels, and complete three-
call timing have no concrete control/layout defect identified. Actual device
results remain decisive. Source and artifact stay frozen; target is free.
