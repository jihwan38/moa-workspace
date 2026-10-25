# Default-off proof

Source34ddf412 includes functional a7c63c30 and nine meaningful real-host lifecycle tests on e71cb50a. a001-recovery-late-library-off:127 passed,0 failed,0 ignored,0 filtered; compile1m32s, tests0.24s. a002-recovery-late-check-off: plain cargo check --bin server succeeded10.16s. Both run records use CARGO_INCREMENTAL=0, package-only debug0 and the agent-owned reused prefill target. All388 snapshotted source/config hashes match the tested tree. Existing unused/dead-code warnings remain.

Independent peer round2_path reviewed both functional hook and all nine tests. Review specifically confirms separate old-decoder removal and after-admission-release compaction under a live512 checkpoint, actual held/EOS/nonstream/error events, receipt bounds and once participation. The only existing production files changed are startup_cohort.rs and worker.rs. The382 other pre-existing src files are byte-identical to e71cb50a, including every model/kernel, Generation/sampling, and fixed90/150 credit policy.

These are CPU policy/ownership checks. No device build, Arena dispatch, or throughput evidence is supplied by this agent. Research activation is separate. Root must freeze/build and evaluate completed matched HTTP runs before adopting; c076 fixed recovery remains held.
