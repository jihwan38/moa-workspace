Source e759c4fcb7dc8e761c310544497cdee175c20549 built release model probe and
server in54.69s via recorded a002-b16-lm-fp8-model-build. All394 source/config
hashes verified unchanged before immutable ELF copies. Kernel matches passed
standalone321121 byte-for-byte. Artifact manifest records both hashes and paths.

Default B16_LM_FP8_ENABLED remainsfalse. No Worker, sampling, cache, wide/scalar/
R32/R64 LM or other kernel selection changed. Static shadow is prepared explicitly
in the probe. The original/B16-FP8 selectors are call-local, and both normalize to
Automatic before existing dispatch guards; failed/missing shadow rejects before
any forward work. Original control stays explicit even for future source activation.

Probe args512 or1024 independently run17 true histories per arm, disjoint physical
slots/pages, original and native B16 output comparisons, widths1/2/8/9/14/16/17,
position0 reset,31-input16+15 narrow reuse,48 disjoint activeP128 interleaves and
full reserved KV equality. Every actual candidate/control B16 greedy result is
checked against its own full logits; old-vs-new greedy/stochastic changes are
reported and do not modify subsequent teacher-forced inputs. C1/R32/prompt outputs
remain strict. Current projection-owned joined FFN storage and group/greedy scratch
are present before measuring startup shadow allocation; preparation is idempotent.

Hostcheck for library, probe and server passed. No full debug-test rebuild was done;
root archived the standalone encoder test's inactive debug family separately.
Root owns Arena/model/quality decisions. The missing-shadow rejection assertion is
specifically an OFF-source probe precondition, so a later ON-source rerun must skip
that assertion or use an explicit unprepared constructor; per-call original control
itself does not depend on the default.
