# Source/hostcheck checkpoint

Frozen functional source: `7915ab3c` (assembly `f1623f8d`, one duplicate module
declaration removed). Host `cargo check --locked --bin test_int4_complete_ffn`
passed in10.88s; captured a002 records accompany this note. a001 is preserved:
an invalid `--features host` command failed before compilation, corrected to
plain cargo check. Original Cargo.toml and frozen ignored Cargo.lock are unchanged.
No device build, resource compile or Arena execution occurred in this tree.

Both independent source reviews passed. e2e checked the seven complete candidate
awaits versus one unchanged production control graph, feedback, timing and counts.
batch checked the independent original P128 BF16 paths, global scaling once,
physical mappings and true row permutation. Root separately verified current
control files against the official q020 source; only loader/helper visibility
changed, with arithmetic bodies unchanged.

Important limits:

- Only supplied layer0 sliding, all512 rows. This does not cover full-prefix or
  padded-tail model routes.
- Step0 drift uses matched initial residuals. Step1 measures cumulative drift
  after each arithmetic route feeds back its own result.
- Correctness success and the10% performance gate are distinct outputs. An
  eventual successful exit alone is not speed evidence; both round gates and
  `I4_COMPLETE_FINAL performance_gate_passed=true` must pass.
- Production weight/norm immutability uses full-read FNV fingerprints, explicitly
  distinguished from the candidate/reference bytewise immutable checks.
- Working-set arithmetic is in `working-set.json`; it is not an allocator or
 48-layer fit result. CPU transformation and original/device startup are charged
  separately in logs, not represented as free model preparation.
- DOWN's245684>240000 resource HOLD remains. No fullchain speed or model quality
  result exists yet.

The next possible gate is a fresh isolated device build followed, only if root
authorizes it, by one complete layer0 execution. See `build-plan.md`. No command
in that document has been executed.
