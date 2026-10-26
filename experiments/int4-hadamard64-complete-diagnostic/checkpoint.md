Complete fixed-H64 FFN diagnostic source checkpoint

Plan33e06e2f; functional70812693; audit-only eb53c35b. All429 prior source/config inputs and dependencies remain unchanged; three new host files make432 inputs. q020/prefix launch bodies match the donor after whitespace and optional trailing-comma normalization. Initial format-only audit false failure is preserved.

CPU a001 passed11 tests and both implementations match all1024 BF16 bits in16 frozen root vectors. Host a002 passed11.16s. Batch independently reviewed all-row provided decode, stage coordinates/RNE, blocked packing, globals and per-row memory. Main/intended-oracle peer review pending.

Candidate uses its OWN GeGLU hidden, then an explicit CPU rotation/readback/upload; production hidden is only a drift reference. Original grouped and q020 arms share each incoming state, and only q020 residual advances step1. Independent P128 boundaries, alternate CPU transform, exact quotient/packing, denseE4-code/typed/raw parity and F64 samples remain strict. All13 scratch states are retained and checked for permutation/reuse; counters use physical owner bytes rather than mixed helper return units.

Original-grouped matched drift anchors fromactual323322 are retained in original-grouped-anchor.json. Future fixed gate: residualL2<=.75original and joined/DOWN no worse, bothorders/steps, positivefinite references. This is complete-FFN numerical progress only; no timing, task-quality or model adoption claim. CPU roundtrips are diagnostic overhead. No device build/Arena has begun.

Build preparation: root independently reviewed the complete main/intended flow without a blocker. One fresh build is conditional on the pending round2 main/oracle PASS. Build input and 19-image comparison audit is prepared. Replay compares all13 field values; residual, rotated_prefix and rotated_hidden are reallocated, so those3 are not a same-allocation reuse claim. No source change was needed for this reporting distinction.

Final round2 main/oracle review PASS at functional70812693/main58b6a85e. Root authorizes one fresh device build after all source gates. Numerical decision remains separate from intended correctness/exit0.

Fresh a003 build PASS54.82s from9c5038a3, capturedELF96ce0228ce20f0afda25206d2bc4ec1e181eb45d9a77e48da759be4d430ce34d (5,686,792B). All432 inputs and original429 preserved.19 images exactly equal both actual323213/323322 and fresh compiler output; q020 image exact. Actual numerical result remains missing; root owns any Arena run.
