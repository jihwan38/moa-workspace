# Fresh standalone build binding

Functional source `394d721e`, source/hostcheck checkpoint `51f8a3a6`.
Root and round2 independently reviewed the complete probe and helper with no
source blocker. The review covers the independent per960 quotient and blocked
encoder, group scale/offset order, unchanged production control, seven awaited
candidate stages, separate reset/readback boundaries, post-reuse owners and
matched-step versus divergent-feedback reporting. It is not a runtime proof.

Root authorized one fresh `test_int4_grouped_complete_ffn` ELF after those
reviews. Original Cargo.toml/lock and the exclusively owned prefill target are
retained. `record_run --fresh-package --capture-artifact` captures the binary
immediately; a newly created empty device OUT_DIR excludes prior cache reuse.
Source remains frozen until capture completes. The target was idle and the
filesystem had 5,272,932,352 bytes available before preparation.

The expected 19 embedded entries must each match an actually validated prior
entry: original complete FFN323067 for unchanged bridge/norm/tail/control,
grouped quantizer323173 for two quantizer entries, and grouped projection323172
for candidate/reference UP/DOWN. The production control must also match the
frozen q020 server image. No new device arithmetic is introduced here.

No Arena execution or model activation is authorized to this agent. Root will
verify captured inputs/artifact/images before any execution. A later whole
chain run retains the >=10% gain gate in each row order; standalone component
timings do not establish that result.
