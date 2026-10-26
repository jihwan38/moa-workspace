Source and CPU checkpoint: functional helper ac69cc26, host diagnostic 4d3dd463,
aggregate-bound plan 48153ee6. Plain hostcheck a003 passed in 11.34 seconds with
427 package inputs. All 423 inherited tracked source/config files and six copied
candidate/control/intended functions remain byte-identical; original Cargo.lock
is also checked. Run `python3 experiments/int4-weight-mse-diagnostic/audit_source.py`
to replay the source and check binding.

Independent reviews: round2_path passed the helper and independent Fraction
proof scope; batch_kernels passed the frozen fixture/main at 4d3dd463. Root
independently passed helper, fixture, matched-feedback flow and both original/
fitted reuse protections. These are source reviews, not device execution proof.
The exact i128 aggregate bound is recorded in plan.json; the independent Fraction
files retain their original hashes. CPU proof results and the preserved initial
tie-counter-only failure are in cpu-result.md and the a001/a002 records.

The diagnostic invokes the validated seven-call chain. It does not invoke the
threegraph entry points present in its base tree. A subsequent approved fresh
build must therefore contain exactly the 19 prior actual323213 device images,
including the original q020 control, and bind every image to its new OUT_DIR.
The base's 22-image threegraph executable is not the expected registry here.

Root authorized that one standalone captured build after this evidence commit.
Source/dependencies are frozen, the prefill target is exclusive, and a new empty
device output directory is mandatory. Root alone may run Arena. No serving or
model activation, performance inference, or task-quality claim is authorized.
