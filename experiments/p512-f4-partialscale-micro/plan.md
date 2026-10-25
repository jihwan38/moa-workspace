# Complete P512 approximate F4 partial-scale gate

Base2e7226c6 preserves dffea370 candidate byte-for-byte. Current control is
ops_group_ffn_projection_storegroup; intended reference changes only F4 weights
to dense unscaled E4 fixed-code weights and removes the lookup. Per960 group
scales stay after each local dot; original activation quantization/norm/GeGLU/
row-global scaling stays. No model or Worker activation.

Provided layers0/5/6/47, full and sliding wrappers, all7 retained buffers,
separately poisoned candidate-first calls, two feedback steps, exact row
permutations, zero256/384 padded rows, independent CPU physical pack and F64
UP/GATE/DOWN samples plus final norm samples. Original E4 weight/stage errors
are reported separately from intended all-bit reference. Each feedback arm
starts from the same prior candidate residual. Input projections are immutable.

Completed-call ABBA compares the current one-launch graph against candidate,
with identical residual reset outside timers. All lookup/partial scale/realDM
movement remains inside. >=10% actual whole-chain gain before host/model work.
Static weight conversion, reference copies and readbacks are untimed; CPU
encoder startup must be separately measured. Root alone dispatches Arena.
